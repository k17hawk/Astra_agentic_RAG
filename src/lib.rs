use std::collections::{HashMap, HashSet, BinaryHeap};
use std::cmp::Ordering;
use unicode_normalization::UnicodeNormalization;
use std::sync::Arc;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::Bound;

// ============================================================================
// Type aliases and basic types
// ============================================================================

type TokenId = usize;
type RootId = usize;
type ByteVal = u8;
type Frequency = u64;

// ============================================================================
// GPT-2 byte<->unicode alphabet (Blocker 5)
// ----------------------------------------------------------------------------
// Every raw byte gets a *printable, single-char* surface. Printable ASCII and
// Latin-1 map to themselves; everything else (control bytes, 0x20 space, 0x7F,
// 0x80..0xA0, 0xAD) maps to an obscure char at 256+n. Space (0x20) is NOT
// special-cased -> it becomes U+0120 'Ġ', which is what enables the Ġ
// space-prefix trick later. Decode never needs the inverse map because byte
// tokens are recovered by *type* (Token::ByteFallback), not by surface.
// ============================================================================

/// Reverse the training driver's TSV escaping (\\ -> \, \t -> tab, \n -> nl).
/// Single left-to-right pass so a real backslash isn't mis-paired with a
/// following t/n.
fn unescape_tsv(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('t') => out.push('\t'),
                Some('n') => out.push('\n'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\u{0}'; 256];
    let mut n: u32 = 0;
    for b in 0u8..=255u8 {
        let code: u32 = if b.is_ascii_graphic()               // 0x21..=0x7E
            || (0xA1u8..=0xAC).contains(&b)                    // ¡..¬
            || (0xAEu8..=0xFF).contains(&b)                    // ®..ÿ
        {
            b as u32
        } else {
            let c = 256 + n;
            n += 1;
            c
        };
        table[b as usize] = char::from_u32(code).expect("valid scalar");
    }
    table
}

// ============================================================================
// Script types (total assignment per §2.1)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Script {
    DEV,
    LAT,
    PUN,
    FMT,
    MAL,
}

impl Script {
    fn rank(&self) -> u8 {
        match self {
            Script::DEV => 2,
            Script::PUN => 1,
            _ => 0,
        }
    }
}

// ============================================================================
// Phase 1: Normalization N  (Blocker 4 — sequence rewriter)
// ----------------------------------------------------------------------------
// N(s) = StripZWJ ∘ Fold_O ∘ NFC(s).
// Fold_O is now a set of string->string rules applied leftmost-longest, so it
// can collapse multi-codepoint sequences (e.g. सङ्ग -> संग) that a char->char
// table cannot. ZWJ (U+200D) is stripped; ZWNJ (U+200C) is preserved.
//
// Idempotence caveat: N(N(s)) = N(s) holds only if every rule's REPLACEMENT is
// already NFC-stable. संग is NFC-stable, so the shipped rules are fine; if you
// add a rule whose output is not NFC-stable, either fix the rule or re-run
// `.nfc()` on the final result.
// ============================================================================

pub struct Normalizer {
    fold_rules: Vec<(Vec<char>, String)>, // (pattern chars, replacement), longest-first
}

impl Normalizer {
    pub fn new(fold_rules: Vec<(String, String)>) -> Self {
        let mut rules: Vec<(Vec<char>, String)> = fold_rules
            .into_iter()
            .map(|(p, r)| (p.chars().collect::<Vec<char>>(), r))
            .collect();
        // Longer patterns first so ङ्ग wins over ङ at the same position.
        rules.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        Self { fold_rules: rules }
    }

    pub fn normalize(&self, s: &str) -> String {
        let nfc: String = s.nfc().collect();
        let chars: Vec<char> = nfc.chars().collect();
        let mut result = String::with_capacity(nfc.len());
        let mut i = 0;

        while i < chars.len() {
            let mut matched = false;
            for (pat, rep) in &self.fold_rules {
                let plen = pat.len();
                if plen > 0 && i + plen <= chars.len() && chars[i..i + plen] == pat[..] {
                    result.push_str(rep);
                    i += plen;
                    matched = true;
                    break;
                }
            }
            if !matched {
                let ch = chars[i];
                if ch != '\u{200D}' {
                    // StripZWJ; ZWNJ (U+200C) is deliberately preserved.
                    result.push(ch);
                }
                i += 1;
            }
        }
        result
    }
}

// ============================================================================
// Phase 2: Akshara DFA
// ----------------------------------------------------------------------------
// (Unchanged. If you later swap this for UAX#29 grapheme segmentation via the
// unicode-segmentation crate, verify the resolved version implements GB9c/InCB
// so conjuncts like क्ष are not split, and test ZWNJ-embedded aksharas.)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Akshara {
    pub surface: String,
    pub root_set: Vec<RootId>,
}

#[derive(Debug, Clone)]
pub struct AksharaDFA {
    pub transitions: HashMap<(usize, char), usize>,
    pub accepting_states: HashSet<usize>,
    initial_state: usize,
}

impl AksharaDFA {
    pub fn new() -> Self {
        Self {
            transitions: HashMap::new(),
            accepting_states: HashSet::new(),
            initial_state: 0,
        }
    }

    /// Debug: check if a transition exists
    pub fn has_transition(&self, state: usize, ch: char) -> Option<usize> {
        self.transitions.get(&(state, ch)).copied()
    }

    /// Debug: get all transitions from a state
    pub fn get_transitions_from(&self, state: usize) -> Vec<(char, usize, bool)> {
        self.transitions
            .iter()
            .filter(|((s, _), _)| *s == state)
            .map(|((_, ch), &next)| (*ch, next, self.accepting_states.contains(&next)))
            .collect()
    }

    /// Tokenize using maximal munch: find the LONGEST valid akshara at each position
    pub fn tokenize(&self, dev_text: &str) -> Vec<Akshara> {
        let mut aksharas = Vec::new();
        let chars: Vec<char> = dev_text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let mut state = self.initial_state;
            let mut last_accepting_pos: Option<usize> = None;
            let mut j = i;

            while j < chars.len() {
                let ch = chars[j];
                if let Some(&next_state) = self.transitions.get(&(state, ch)) {
                    state = next_state;
                    if self.accepting_states.contains(&state) {
                        last_accepting_pos = Some(j + 1);
                    }
                    j += 1;
                } else {
                    break;
                }
            }

            if let Some(end) = last_accepting_pos {
                let akshara_str: String = chars[i..end].iter().collect();
                aksharas.push(Akshara {
                    surface: akshara_str,
                    root_set: Vec::new(), // populated later via ParadigmRegistry (§2 tagging)
                });
                i = end;
            } else {
                let ch = chars[i];
                aksharas.push(Akshara {
                    surface: ch.to_string(),
                    root_set: Vec::new(),
                });
                i += 1;
            }
        }
        aksharas
    }
}

// ============================================================================
// Token types and vocabulary
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Token {
    Akshara(Akshara),
    Punctuation(String),
    ZWNJ,
    ByteFallback(ByteVal),
    SeededMorpheme(String),
    MergedToken(Vec<TokenId>),
    /// Latin/ASCII alphanumeric base unit (Phase 4). These MUST exist as their
    /// own script class: if 'h' were only a ByteFallback token its script would
    /// be MAL, and the Gate's atomic-class clause would forbid every Latin
    /// merge. Seeded before the byte alphabet so it wins the surface key.
    Latin(String),
    /// Reconstructed from a saved vocab (surface only, no merge/child history).
    /// Enough for encode + decode; NOT enough to correctly resume training.
    Loaded(String),
}

impl Token {
    /// Default script by variant. NOTE: for MergedToken this is only a
    /// placeholder — Vocabulary::create_merged overrides id_to_script with the
    /// left child's actual script, and Vocabulary::get_script is authoritative.
    /// For Loaded tokens the script is set directly during load, not from here.
    pub fn script(&self) -> Script {
        match self {
            Token::Akshara(_) => Script::DEV,
            Token::Punctuation(_) => Script::PUN,
            Token::ZWNJ => Script::FMT,
            Token::ByteFallback(_) => Script::MAL,
            Token::SeededMorpheme(_) => Script::DEV,
            Token::MergedToken(_) => Script::DEV,
            Token::Latin(_) => Script::LAT,
            Token::Loaded(_) => Script::DEV,
        }
    }
}

// ============================================================================
// Vocabulary management (surface string as primary key)
// ============================================================================

#[derive(Default)]
pub struct Vocabulary {
    tokens: Vec<Arc<Token>>,
    surface_to_id: HashMap<String, TokenId>,
    id_to_script: HashMap<TokenId, Script>,
    v_strict: HashSet<TokenId>,
    v_ambiguous: HashSet<TokenId>,
    token_to_root_set: HashMap<TokenId, Vec<RootId>>,
    surfaces: HashMap<TokenId, String>,
    /// Longest surface in CHARS — the cap for greedy longest-match encoding.
    max_surface_len: usize,
}

impl Vocabulary {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn initialize(
        &mut self,
        base_aksharas: Vec<Akshara>,
        seed_morphemes: Vec<String>,
        punctuation: Vec<String>,
        v_strict: HashSet<String>,
        v_ambiguous: HashSet<String>,
        byte_encoder: &[char; 256],
    ) {
        for akshara in base_aksharas {
            let surface = akshara.surface.clone();
            let token = Arc::new(Token::Akshara(akshara));
            self.add_token(token, surface, false, false);
        }
        for morph in seed_morphemes {
            let surface = morph.clone();
            let is_strict = v_strict.contains(&morph);
            let is_ambiguous = v_ambiguous.contains(&morph);
            let token = Arc::new(Token::SeededMorpheme(morph));
            self.add_token(token, surface, is_strict, is_ambiguous);
        }
        for punct in punctuation {
            let surface = punct.clone();
            let token = Arc::new(Token::Punctuation(punct));
            self.add_token(token, surface, false, false);
        }
        // Phase 4 base alphabet: Latin letters + ASCII digits as LAT-scripted
        // tokens. Seeded BEFORE the byte alphabet so these surfaces resolve to
        // LAT (mergeable) rather than MAL byte tokens (never mergeable).
        for ch in ('a'..='z').chain('A'..='Z').chain('0'..='9') {
            let surface = ch.to_string();
            let token = Arc::new(Token::Latin(surface.clone()));
            self.add_token(token, surface, false, false);
        }

        // Byte-fallback alphabet, GPT-2 style. Punctuation and Latin are added
        // *before* this, so an ASCII-graphic byte that collides with an existing
        // surface aliases to that token; harmless — the surface still decodes
        // correctly, and single-byte ASCII never appears as a UTF-8 continuation
        // byte, so the multibyte flush is unaffected.
        for byte_val in 0u8..=255 {
            let surface = byte_encoder[byte_val as usize].to_string();
            let token = Arc::new(Token::ByteFallback(byte_val));
            self.add_token(token, surface, false, false);
        }
        let zwnj = Arc::new(Token::ZWNJ);
        self.add_token(zwnj, "\u{200C}".to_string(), false, false);
    }

    fn add_token(
        &mut self,
        token: Arc<Token>,
        surface: String,
        is_strict: bool,
        is_ambiguous: bool,
    ) -> TokenId {
        if let Some(&existing_id) = self.surface_to_id.get(&surface) {
            return existing_id;
        }

        let id = self.tokens.len();
        let script = token.script();
        let clen = surface.chars().count();
        if clen > self.max_surface_len {
            self.max_surface_len = clen;
        }
        self.surface_to_id.insert(surface.clone(), id);
        self.id_to_script.insert(id, script);
        self.tokens.push(token);
        self.surfaces.insert(id, surface);

        if is_strict {
            self.v_strict.insert(id);
        }
        if is_ambiguous {
            self.v_ambiguous.insert(id);
        }
        if let Token::Akshara(a) = &*self.tokens[id] {
            self.token_to_root_set.insert(id, a.root_set.clone());
        }
        id
    }

    /// §2 tagging: RootSet(α) = { root : α ∈ prefixes(P(root)) }.
    /// Call AFTER paradigms are loaded and BEFORE training. Costs nothing
    /// behaviorally if no paradigms are registered (all root sets stay empty,
    /// Morph stays in its RootSet=∅ ⇒ 1 branch). This is the wire that makes the
    /// paradigm machinery non-inert once a real FST populates the registry.
    pub fn assign_roots_from_registry(&mut self, registry: &ParadigmRegistry) {
        for id in 0..self.tokens.len() {
            let eligible = matches!(
                &*self.tokens[id],
                Token::Akshara(_) | Token::SeededMorpheme(_) | Token::MergedToken(_)
            );
            if !eligible {
                continue;
            }
            if let Some(surface) = self.surfaces.get(&id).cloned() {
                let roots = registry.get_root_set(&surface);
                if !roots.is_empty() {
                    self.token_to_root_set.insert(id, roots);
                }
            }
        }
    }

    pub fn get_id_by_surface(&self, surface: &str) -> Option<TokenId> {
        self.surface_to_id.get(surface).copied()
    }

    pub fn max_surface_len(&self) -> usize {
        self.max_surface_len
    }

    /// Rebuild the vocabulary from saved (id, surface) pairs — enough for
    /// encode + decode. Byte tokens are recovered as ByteFallback via the byte
    /// alphabet; everything else becomes a Loaded surface token. v_strict /
    /// v_ambiguous / root sets are NOT restored, so a loaded vocab can tokenize
    /// but should not be used to resume training.
    ///
    /// `pairs` must have contiguous ids 0..N; they are sorted defensively.
    pub fn load_from_pairs(&mut self, mut pairs: Vec<(TokenId, String)>, byte_decoder: &HashMap<char, u8>) {
        self.tokens.clear();
        self.surface_to_id.clear();
        self.id_to_script.clear();
        self.v_strict.clear();
        self.v_ambiguous.clear();
        self.token_to_root_set.clear();
        self.surfaces.clear();
        self.max_surface_len = 0;

        pairs.sort_by_key(|(id, _)| *id);

        for (expected_id, surface) in pairs {
            let id = self.tokens.len();
            debug_assert_eq!(id, expected_id, "vocab ids must be contiguous from 0");

            // Classify the surface. Order matters: Latin/digit chars are checked
            // BEFORE the byte alphabet, because 'a' is both a valid Latin base
            // token and byte 0x61 — it must come back as LAT (mergeable), not
            // MAL. Merged surfaces are always >= 2 chars, so the single-char
            // checks never misfire on a real merge.
            let mut token = Arc::new(Token::Loaded(surface.clone()));
            let mut chs = surface.chars();
            if let (Some(ch), None) = (chs.next(), chs.next()) {
                if ch.is_ascii_alphanumeric() {
                    token = Arc::new(Token::Latin(surface.clone()));
                } else if let Some(&b) = byte_decoder.get(&ch) {
                    token = Arc::new(Token::ByteFallback(b));
                }
            } else if surface.chars().all(|c| c.is_ascii_alphanumeric()) {
                // Multi-char pure-ASCII surface = a Latin merge from Phase 4.
                token = Arc::new(Token::Latin(surface.clone()));
            }

            let clen = surface.chars().count();
            if clen > self.max_surface_len {
                self.max_surface_len = clen;
            }
            let script = token.script();
            self.surface_to_id.insert(surface.clone(), id);
            self.id_to_script.insert(id, script);
            self.surfaces.insert(id, surface);
            self.tokens.push(token);
        }
    }

    pub fn get_token(&self, id: TokenId) -> Option<&Arc<Token>> {
        self.tokens.get(id)
    }

    pub fn get_script(&self, id: TokenId) -> Script {
        *self.id_to_script.get(&id).unwrap_or(&Script::MAL)
    }

    pub fn is_strict(&self, id: TokenId) -> bool {
        self.v_strict.contains(&id)
    }

    pub fn is_ambiguous(&self, id: TokenId) -> bool {
        self.v_ambiguous.contains(&id)
    }

    pub fn get_root_set(&self, id: TokenId) -> &[RootId] {
        self.token_to_root_set
            .get(&id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Blocker 3: a merged token inherits the LEFT child's script (ScriptCompat
    /// guarantees both children share it). Guarded so that if `merged_surface`
    /// already exists, we return the existing id WITHOUT clobbering its script
    /// or root set.
    pub fn create_merged(&mut self, a: TokenId, b: TokenId, merged_root_set: Vec<RootId>) -> TokenId {
        let surface_a = self.get_surface(a).unwrap_or_default();
        let surface_b = self.get_surface(b).unwrap_or_default();
        let merged_surface = format!("{}{}", surface_a, surface_b);

        let existed = self.surface_to_id.contains_key(&merged_surface);
        let merged = Arc::new(Token::MergedToken(vec![a, b]));
        let id = self.add_token(merged, merged_surface, false, false);

        if !existed {
            if let Some(&script) = self.id_to_script.get(&a) {
                self.id_to_script.insert(id, script);
            }
            self.token_to_root_set.insert(id, merged_root_set);
        }
        id
    }

    pub fn get_surface(&self, id: TokenId) -> Option<String> {
        self.surfaces.get(&id).cloned()
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Debug: get all surfaces
    pub fn get_all_surfaces(&self) -> Vec<(TokenId, String)> {
        self.surfaces.iter().map(|(&id, s)| (id, s.clone())).collect()
    }

    /// Debug: check if surface exists
    pub fn contains_surface(&self, surface: &str) -> bool {
        self.surface_to_id.contains_key(surface)
    }
}

// ============================================================================
// Paradigm morphology (FST-backed)
// ============================================================================

pub struct Paradigm {
    pub root_id: RootId,
    pub allowed_transitions: HashMap<String, HashSet<String>>,
}

impl Paradigm {
    pub fn new(root_id: RootId) -> Self {
        Self {
            root_id,
            allowed_transitions: HashMap::new(),
        }
    }

    pub fn allows(&self, prefix: &str, continuation: &str) -> bool {
        self.allowed_transitions
            .get(prefix)
            .map(|allowed| allowed.contains(continuation))
            .unwrap_or(false)
    }

    pub fn has_prefix(&self, prefix: &str) -> bool {
        self.allowed_transitions.contains_key(prefix)
    }
}

#[derive(Default)]
pub struct ParadigmRegistry {
    paradigms: Vec<Paradigm>,
    root_to_paradigm: HashMap<RootId, usize>,
}

impl ParadigmRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_paradigm(&mut self, paradigm: Paradigm) {
        let idx = self.paradigms.len();
        self.root_to_paradigm.insert(paradigm.root_id, idx);
        self.paradigms.push(paradigm);
    }

    pub fn get_root_set(&self, prefix: &str) -> Vec<RootId> {
        self.paradigms
            .iter()
            .filter(|p| p.has_prefix(prefix))
            .map(|p| p.root_id)
            .collect()
    }

    pub fn check_allowed(&self, root_id: RootId, prefix: &str, continuation: &str) -> bool {
        if let Some(&idx) = self.root_to_paradigm.get(&root_id) {
            self.paradigms[idx].allows(prefix, continuation)
        } else {
            false
        }
    }
}

// ============================================================================
// Phase 3: Constrained BPE
// ============================================================================

/// Decrement a pair's count by `by`; prune the entry entirely when it reaches 0
/// so that pair_freqs.keys() never iterates dead entries.
fn dec_pair(freqs: &mut HashMap<(TokenId, TokenId), Frequency>, key: (TokenId, TokenId), by: Frequency) {
    if let Some(f) = freqs.get_mut(&key) {
        *f = f.saturating_sub(by);
        if *f == 0 {
            freqs.remove(&key);
        }
    }
}

fn inc_pair(freqs: &mut HashMap<(TokenId, TokenId), Frequency>, key: (TokenId, TokenId), by: Frequency) {
    *freqs.entry(key).or_insert(0) += by;
}

/// One deduplicated word TYPE: its initial token sequence and how many times it
/// occurred in the corpus. This is the RAM + speed fix — the trainer iterates a
/// few million unique word types, not billions of raw token positions, and pair
/// frequencies are weighted by `count`.
struct Word {
    tokens: Vec<TokenId>,
    count: Frequency,
}

pub struct Corpus {
    words: Vec<Word>,
    pair_freqs: HashMap<(TokenId, TokenId), Frequency>,
    vocab_budget: usize,
}

impl Corpus {
    /// Build from a word-frequency dictionary (streaming path). Consumes `counts`.
    pub fn from_word_counts(counts: HashMap<Vec<TokenId>, Frequency>, vocab_budget: usize) -> Self {
        let words: Vec<Word> = counts
            .into_iter()
            .map(|(tokens, count)| Word { tokens, count })
            .collect();
        let mut corpus = Self {
            words,
            pair_freqs: HashMap::new(),
            vocab_budget,
        };
        corpus.recompute_all_frequencies();
        corpus
    }

    /// Build from raw sequences with no dedup (each sequence has count 1).
    /// Kept for the in-memory `train_bpe` / `train_from_text` path; do NOT use
    /// this for very large corpora — use `from_word_counts` via `train_from_file`.
    pub fn from_sequences(sequences: Vec<Vec<TokenId>>, vocab_budget: usize) -> Self {
        let words: Vec<Word> = sequences
            .into_iter()
            .map(|tokens| Word { tokens, count: 1 })
            .collect();
        let mut corpus = Self {
            words,
            pair_freqs: HashMap::new(),
            vocab_budget,
        };
        corpus.recompute_all_frequencies();
        corpus
    }

    /// Full scan — used ONCE at construction only. Weighted by word count.
    fn recompute_all_frequencies(&mut self) {
        self.pair_freqs.clear();
        for w in &self.words {
            for window in w.tokens.windows(2) {
                *self.pair_freqs
                    .entry((window[0], window[1]))
                    .or_insert(0) += w.count;
            }
        }
    }

    pub fn get_freq(&self, a: TokenId, b: TokenId) -> Frequency {
        self.pair_freqs.get(&(a, b)).copied().unwrap_or(0)
    }

    /// Blocker 6: incremental merge. Applies (a,b)->new_id across the corpus and
    /// updates pair frequencies with local deltas only (no full recompute).
    /// Returns the deduplicated set of AFFECTED pairs — both the newly formed
    /// pairs AND the decremented neighbours — so the caller re-checks Legal and
    /// pushes a fresh heap entry for each. Re-pushing the *decremented*
    /// neighbours is the fix for the "legal pair silently vanishes" bug: their
    /// old snapshots are now stale and would be discarded on pop with nothing to
    /// replace them.
    ///
    /// Borrow note: `self.words.iter_mut()` and `&mut self.pair_freqs` are
    /// disjoint fields, so the split borrow compiles. Keep the freq mutation as
    /// free functions (dec_pair/inc_pair) — routing it through a `&mut self`
    /// method would re-borrow all of `self` and fail to compile.
    pub fn apply_merge(&mut self, a: TokenId, b: TokenId, new_id: TokenId) -> Vec<(TokenId, TokenId)> {
        let mut touched: HashSet<(TokenId, TokenId)> = HashSet::new();

        for w in self.words.iter_mut() {
            let cnt = w.count;
            let mut i = 0;
            while i + 1 < w.tokens.len() {
                if w.tokens[i] == a && w.tokens[i + 1] == b {
                    // Destroy old neighbour pairs (weighted by this word's count).
                    if i > 0 {
                        let l = w.tokens[i - 1];
                        dec_pair(&mut self.pair_freqs, (l, a), cnt);
                        touched.insert((l, a));
                    }
                    if i + 2 < w.tokens.len() {
                        let r = w.tokens[i + 2];
                        dec_pair(&mut self.pair_freqs, (b, r), cnt);
                        touched.insert((b, r));
                    }

                    // Consume (a, b) -> new_id.
                    w.tokens[i] = new_id;
                    w.tokens.remove(i + 1);

                    // Form new neighbour pairs.
                    if i > 0 {
                        let l = w.tokens[i - 1];
                        inc_pair(&mut self.pair_freqs, (l, new_id), cnt);
                        touched.insert((l, new_id));
                    }
                    if i + 1 < w.tokens.len() {
                        let r = w.tokens[i + 1];
                        inc_pair(&mut self.pair_freqs, (new_id, r), cnt);
                        touched.insert((new_id, r));
                    }
                    // Do NOT advance i: staying lets an overlapping (a,b) newly
                    // beginning at i be caught.
                } else {
                    i += 1;
                }
            }
        }

        // (a, b) is fully consumed corpus-wide.
        self.pair_freqs.remove(&(a, b));
        touched.into_iter().collect()
    }
}

#[derive(Debug, Clone)]
struct MergeCandidate {
    a: TokenId,
    b: TokenId,
    priority_key: (u8, u64),
    freq_snapshot: Frequency,
}

impl PartialEq for MergeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.priority_key == other.priority_key && self.a == other.a && self.b == other.b
    }
}
impl Eq for MergeCandidate {}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Blocker 1: NO .reverse(). BinaryHeap is a max-heap, so this pops the LARGEST
// (ScriptRank, freq) first = §3.3 argmax. Ties are broken deterministically by
// (a, b) so trained vocabularies are reproducible, and Ord is consistent with
// the PartialEq above.
impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority_key
            .cmp(&other.priority_key)
            .then_with(|| self.a.cmp(&other.a))
            .then_with(|| self.b.cmp(&other.b))
    }
}

pub struct ConstrainedBPETrainer {
    vocab: Vocabulary,
    paradigm_registry: ParadigmRegistry,
    pub theta: Frequency,
    /// PHASE 4. When true this is the Latin secondary pass: unconstrained BPE
    /// over LAT runs only. No V_seed/V_strict/V_ambiguous (those are Devanagari
    /// morphemes) and no Morph (no paradigms for English) — pure frequency.
    /// When false it is the Phase-3 constrained Devanagari pass, unchanged.
    pub latin_pass: bool,
}

impl ConstrainedBPETrainer {
    pub fn new(vocab: Vocabulary, paradigm_registry: ParadigmRegistry) -> Self {
        Self {
            vocab,
            paradigm_registry,
            theta: 100,
            latin_pass: false,
        }
    }

    fn script_compat(&self, a: TokenId, b: TokenId) -> bool {
        let sa = self.vocab.get_script(a);
        let sb = self.vocab.get_script(b);
        if self.latin_pass {
            // Phase 4: Latin only. Fully separate from the DEV vocabulary, so no
            // DEV/LAT token can ever form (the v2 "no token spans two scripts"
            // guarantee still holds).
            return sa == sb && sa == Script::LAT;
        }
        sa == sb && (sa == Script::DEV || sa == Script::PUN)
    }

    fn gate(&self, a: TokenId, b: TokenId, freq: Frequency) -> bool {
        let sa = self.vocab.get_script(a);
        let sb = self.vocab.get_script(b);

        if sa == Script::MAL || sa == Script::FMT || sb == Script::MAL || sb == Script::FMT {
            return false;
        }
        if !self.script_compat(a, b) {
            return false;
        }
        if self.latin_pass {
            return true; // unconstrained: frequency alone decides
        }
        if self.vocab.is_strict(b) || self.vocab.is_ambiguous(b) {
            return false;
        }
        if self.vocab.is_strict(a) {
            return false;
        }
        if self.vocab.is_ambiguous(a) && freq < self.theta {
            return false;
        }
        true
    }

    fn morph(&self, a: TokenId, b: TokenId) -> bool {
        if self.latin_pass {
            return true; // no paradigms for Latin
        }
        let root_set = self.vocab.get_root_set(a);
        if root_set.is_empty() {
            return true; // non-paradigm token: defer to Gate + Freq (§3.2 fix #2)
        }
        let surface_a = self.vocab.get_surface(a).unwrap_or_default();
        let surface_b = self.vocab.get_surface(b).unwrap_or_default();

        root_set
            .iter()
            .any(|&root_id| self.paradigm_registry.check_allowed(root_id, &surface_a, &surface_b))
    }

    fn legal(&self, a: TokenId, b: TokenId, freq: Frequency) -> bool {
        self.gate(a, b, freq) && self.morph(a, b)
    }

    fn narrow_root_set(&self, a: TokenId, b: TokenId) -> Vec<RootId> {
        if self.latin_pass {
            return Vec::new();
        }
        let root_set_a = self.vocab.get_root_set(a);
        if root_set_a.is_empty() {
            return Vec::new();
        }
        let surface_a = self.vocab.get_surface(a).unwrap_or_default();
        let surface_b = self.vocab.get_surface(b).unwrap_or_default();

        root_set_a
            .iter()
            .filter(|&&root_id| {
                self.paradigm_registry
                    .check_allowed(root_id, &surface_a, &surface_b)
            })
            .copied()
            .collect()
    }

    fn priority_key(&self, a: TokenId, _b: TokenId, freq: Frequency) -> (u8, u64) {
        (self.vocab.get_script(a).rank(), freq)
    }

    /// Train until `budget` (total vocab size) is reached or no admissible merge
    /// remains. `progress_every` merges, prints a timing line to stderr (0=off).
    /// The budget is a TOTAL vocab-size target, not a merge count — so for the
    /// Phase-4 pass you pass (dev_budget + lat_budget).
    pub fn train(&mut self, corpus: &mut Corpus, budget: usize, progress_every: u64) {
        let t0 = std::time::Instant::now();
        let start_vocab = self.vocab.len();
        let tag = if self.latin_pass { "LAT" } else { "DEV" };

        let mut heap: BinaryHeap<MergeCandidate> = BinaryHeap::new();
        self.initialize_heap(corpus, &mut heap);
        if progress_every > 0 {
            eprintln!(
                "[train:{}] heap seeded with {} admissible pairs in {:.1}s",
                tag,
                heap.len(),
                t0.elapsed().as_secs_f64()
            );
        }

        let mut merges: u64 = 0;

        while self.vocab.len() < budget {
            let mut applied = false;

            while let Some(candidate) = heap.pop() {
                let current_freq = corpus.get_freq(candidate.a, candidate.b);

                // §3.4 lazy invalidation.
                if current_freq != candidate.freq_snapshot {
                    continue;
                }
                if current_freq == 0 {
                    continue;
                }
                if !self.legal(candidate.a, candidate.b, current_freq) {
                    continue;
                }

                // Mint the merged token (mutable vocab borrow ends here).
                let narrowed = self.narrow_root_set(candidate.a, candidate.b);
                let new_id = self.vocab.create_merged(candidate.a, candidate.b, narrowed);

                // Incremental corpus update -> only the affected pairs.
                let touched = corpus.apply_merge(candidate.a, candidate.b, new_id);

                // Re-check Legal and push a fresh entry for every affected pair.
                for (x, y) in touched {
                    let f = corpus.get_freq(x, y);
                    if f > 0 && self.legal(x, y, f) {
                        heap.push(MergeCandidate {
                            a: x,
                            b: y,
                            priority_key: self.priority_key(x, y, f),
                            freq_snapshot: f,
                        });
                    }
                }

                merges += 1;
                applied = true;

                if progress_every > 0 && merges % progress_every == 0 {
                    let secs = t0.elapsed().as_secs_f64();
                    eprintln!(
                        "[train:{}] {} merges | vocab {} | heap {} | {:.1}s | {:.0} merges/s",
                        tag,
                        merges,
                        self.vocab.len(),
                        heap.len(),
                        secs,
                        merges as f64 / secs.max(1e-9)
                    );
                }
                break;
            }

            if !applied {
                break; // heap exhausted of admissible pairs
            }
        }

        if progress_every > 0 {
            eprintln!(
                "[train:{}] done: {} merges ({} -> {} vocab) in {:.1}s",
                tag,
                merges,
                start_vocab,
                self.vocab.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }

    fn initialize_heap(&self, corpus: &Corpus, heap: &mut BinaryHeap<MergeCandidate>) {
        for &(a, b) in corpus.pair_freqs.keys() {
            let freq = corpus.get_freq(a, b);
            if self.legal(a, b, freq) {
                heap.push(MergeCandidate {
                    a,
                    b,
                    priority_key: self.priority_key(a, b, freq),
                    freq_snapshot: freq,
                });
            }
        }
    }
}

// ============================================================================
// Main tokenizer
// ============================================================================

#[allow(dead_code)]
pub struct NepBPETokenizer {
    pub normalizer: Normalizer,
    pub akshara_dfa: AksharaDFA,
    pub vocab: Vocabulary,
    pub paradigm_registry: ParadigmRegistry,
    byte_encoder: [char; 256],
    trainer: Option<ConstrainedBPETrainer>,
    paradigm_embedding: HashMap<TokenId, Option<RootId>>,
}

impl NepBPETokenizer {
    pub fn new(fold_rules: Vec<(String, String)>, paradigm_registry: ParadigmRegistry) -> Self {
        Self {
            normalizer: Normalizer::new(fold_rules),
            akshara_dfa: AksharaDFA::new(),
            vocab: Vocabulary::new(),
            paradigm_registry,
            byte_encoder: bytes_to_unicode(),
            trainer: None,
            paradigm_embedding: HashMap::new(),
        }
    }

    pub fn encode(&self, s: &str) -> Vec<TokenId> {
        let normalized = self.normalizer.normalize(s);
        self.encode_normalized(&normalized)
    }

    /// Rebuild the vocab from saved (id, surface) pairs. Encode/decode-ready;
    /// not training-ready (strict/ambiguous/roots are not restored).
    pub fn load_vocab(&mut self, pairs: Vec<(TokenId, String)>) {
        // Invert the byte alphabet: surface char -> byte value.
        let mut byte_decoder: HashMap<char, u8> = HashMap::with_capacity(256);
        for (b, &ch) in self.byte_encoder.iter().enumerate() {
            byte_decoder.insert(ch, b as u8);
        }
        self.vocab.load_from_pairs(pairs, &byte_decoder);
    }

    /// Tokenize text that is ALREADY normalized (N applied). The streaming
    /// trainer normalizes a whole line once, then calls this per whitespace word
    /// — avoiding one NFC/fold pass per word (billions of them at 12.8 GB).
    pub fn encode_normalized(&self, normalized: &str) -> Vec<TokenId> {
        let mut tokens = Vec::new();
        let mut current_run = String::new();
        let mut current_script: Option<Script> = None;

        for ch in normalized.chars() {
            let ch_script = self.classify_char(ch);

            match current_script {
                Some(cs) if cs == ch_script => {
                    current_run.push(ch);
                }
                _ => {
                    if !current_run.is_empty() {
                        if let Some(cs) = current_script {
                            self.tokenize_run(&current_run, cs, &mut tokens);
                        }
                        current_run.clear();
                    }
                    current_run.push(ch);
                    current_script = Some(ch_script);
                }
            }
        }

        if !current_run.is_empty() {
            if let Some(cs) = current_script {
                self.tokenize_run(&current_run, cs, &mut tokens);
            }
        }

        tokens
    }

    fn classify_char(&self, ch: char) -> Script {
        if ch == '\u{200C}' {
            Script::FMT
        } else if ch.is_ascii_punctuation() || ch == '\u{0964}' || ch == '\u{0965}' {
            Script::PUN
        } else if ch.is_ascii_alphabetic() {
            Script::LAT
        } else if ('\u{0900}'..='\u{097F}').contains(&ch) {
            Script::DEV
        } else {
            Script::MAL
        }
    }

    fn tokenize_run(&self, run: &str, script: Script, tokens: &mut Vec<TokenId>) {
        match script {
            // DEV now uses greedy longest-match over the vocab, so the learned
            // merges are actually applied at encode time. (The old code emitted
            // one token per codepoint via the empty DFA and never consulted the
            // merged surfaces — that was the ~5.6-tokens/word bug.) When the DFA
            // is populated you may instead want to segment to aksharas first and
            // greedy-match at akshara granularity to keep the hard conjunct
            // guarantee; char-level greedy is what matches the trained vocab.
            Script::DEV | Script::LAT => self.tokenize_greedy(run, tokens),
            Script::PUN => {
                for ch in run.chars() {
                    if let Some(id) = self.vocab.get_id_by_surface(&ch.to_string()) {
                        tokens.push(id);
                    } else {
                        self.emit_byte_fallback(ch.to_string().as_bytes(), tokens);
                    }
                }
            }
            Script::FMT => {
                if let Some(id) = self.vocab.get_id_by_surface("\u{200C}") {
                    tokens.push(id);
                }
            }
            Script::MAL => {
                self.emit_byte_fallback(run.as_bytes(), tokens);
            }
        }
    }

    /// Greedy longest-match over vocab surfaces, capped at the longest known
    /// surface length; misses fall through to per-byte fallback. Used for both
    /// DEV and LAT runs.
    fn tokenize_greedy(&self, run: &str, tokens: &mut Vec<TokenId>) {
        let chars: Vec<char> = run.chars().collect();
        let n = chars.len();
        let cap = self.vocab.max_surface_len().max(1);
        let mut i = 0;
        while i < n {
            let hi = n.min(i + cap);
            let mut matched = false;
            for j in (i + 1..=hi).rev() {
                let candidate: String = chars[i..j].iter().collect();
                if let Some(id) = self.vocab.get_id_by_surface(&candidate) {
                    tokens.push(id);
                    i = j;
                    matched = true;
                    break;
                }
            }
            if !matched {
                // Single char not in vocab -> spell it out in byte fallback.
                let ch_str = chars[i].to_string();
                self.emit_byte_fallback(ch_str.as_bytes(), tokens);
                i += 1;
            }
        }
    }

    /// Emit one byte-fallback token per raw byte, using the GPT-2 alphabet.
    fn emit_byte_fallback(&self, bytes: &[u8], tokens: &mut Vec<TokenId>) {
        for &byte in bytes {
            let surface = self.byte_encoder[byte as usize].to_string();
            if let Some(id) = self.vocab.get_id_by_surface(&surface) {
                tokens.push(id);
            }
        }
    }

    /// Blocker 2 + decode-by-type: reconstruct bytes from ByteFallback tokens
    /// (detected by TYPE, not surface, so it is independent of the byte
    /// alphabet), and preserve ZWNJ verbatim (NO surface skip). Flushes buffered
    /// bytes with from_utf8_lossy so a model-generated arbitrary byte stream
    /// degrades gracefully instead of dropping a whole run.
    pub fn decode(&self, token_ids: &[TokenId]) -> String {
        let mut result = String::new();
        let mut byte_buf: Vec<u8> = Vec::new();

        for &id in token_ids {
            match self.vocab.get_token(id).map(|arc| arc.as_ref()) {
                Some(Token::ByteFallback(byte)) => {
                    byte_buf.push(*byte);
                }
                Some(_) => {
                    if !byte_buf.is_empty() {
                        result.push_str(&String::from_utf8_lossy(&std::mem::take(&mut byte_buf)));
                    }
                    if let Some(surface) = self.vocab.get_surface(id) {
                        result.push_str(&surface); // ZWNJ included, like any other surface
                    }
                }
                None => {
                    if !byte_buf.is_empty() {
                        result.push_str(&String::from_utf8_lossy(&std::mem::take(&mut byte_buf)));
                    }
                }
            }
        }

        if !byte_buf.is_empty() {
            result.push_str(&String::from_utf8_lossy(&byte_buf));
        }
        result
    }

    pub fn verify_roundtrip(&self, s: &str) -> bool {
        let encoded = self.encode(s);
        let decoded = self.decode(&encoded);
        decoded == self.normalizer.normalize(s)
    }
}

// ============================================================================
// Python bindings
// ============================================================================

#[pyclass]
pub struct PyNepBPETokenizer {
    inner: NepBPETokenizer,
}

#[pymethods]
impl PyNepBPETokenizer {
    /// Blocker 4: pass folding rules as a list of (pattern, replacement) string
    /// pairs, e.g. [("सङ्ग", "संग"), ("सँग", "संग")], instead of a char->char dict.
    #[new]
    #[pyo3(signature = (folding_rules=None))]
    fn new(folding_rules: Option<Vec<(String, String)>>) -> PyResult<Self> {
        let rules = folding_rules.unwrap_or_default();
        let registry = ParadigmRegistry::new();
        let inner = NepBPETokenizer::new(rules, registry);
        Ok(Self { inner })
    }

    fn normalize(&self, s: &str) -> String {
        self.inner.normalizer.normalize(s)
    }

    fn encode(&self, text: &str) -> PyResult<Vec<usize>> {
        Ok(self.inner.encode(text))
    }

    fn decode(&self, ids: Vec<usize>) -> String {
        self.inner.decode(&ids)
    }

    fn initialize_vocab(
        &mut self,
        aksharas: Vec<String>,
        seed_morphemes: Vec<String>,
        punctuation: Vec<String>,
        v_strict: Vec<String>,
        v_ambiguous: Vec<String>,
    ) -> PyResult<usize> {
        let base_aksharas: Vec<Akshara> = aksharas
            .into_iter()
            .map(|s| Akshara {
                surface: s,
                root_set: Vec::new(),
            })
            .collect();

        self.inner.vocab.initialize(
            base_aksharas,
            seed_morphemes,
            punctuation,
            v_strict.into_iter().collect(),
            v_ambiguous.into_iter().collect(),
            &self.inner.byte_encoder,
        );

        Ok(self.inner.vocab.len())
    }

    /// Explicitly tag base/seed tokens with paradigm roots. Also called
    /// automatically at the start of training; safe to call more than once.
    fn assign_initial_roots(&mut self) {
        self.inner
            .vocab
            .assign_roots_from_registry(&self.inner.paradigm_registry);
    }

    fn add_dfa_transition(&mut self, state: usize, ch: char, next_state: usize, accepting: bool) {
        self.inner
            .akshara_dfa
            .transitions
            .insert((state, ch), next_state);
        if accepting {
            self.inner.akshara_dfa.accepting_states.insert(next_state);
        }
    }

    /// Debug: Check if a DFA transition exists
    fn dfa_has_transition(&self, state: usize, ch: char) -> bool {
        self.inner.akshara_dfa.has_transition(state, ch).is_some()
    }

    /// Debug: Get transitions from a state
    fn dfa_get_transitions(&self, state: usize) -> Vec<(String, usize, bool)> {
        self.inner
            .akshara_dfa
            .get_transitions_from(state)
            .into_iter()
            .map(|(ch, next, acc)| (ch.to_string(), next, acc))
            .collect()
    }

    /// Debug: Test DFA tokenization
    fn dfa_tokenize_debug(&self, text: &str) -> Vec<String> {
        self.inner
            .akshara_dfa
            .tokenize(text)
            .into_iter()
            .map(|a| a.surface)
            .collect()
    }

    /// Debug: Check if surface exists in vocab
    fn vocab_contains(&self, surface: &str) -> bool {
        self.inner.vocab.contains_surface(surface)
    }

    /// Debug: Get token ID for surface
    fn vocab_get_id(&self, surface: &str) -> Option<usize> {
        self.inner.vocab.get_id_by_surface(surface)
    }

    /// Debug: Get surface for token ID
    fn vocab_get_surface(&self, id: usize) -> PyResult<String> {
        self.inner
            .vocab
            .get_surface(id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid token ID"))
    }

    fn add_paradigm(&mut self, root_id: usize, transitions: Bound<'_, PyDict>) -> PyResult<()> {
        let mut paradigm = Paradigm::new(root_id);
        for (prefix, continuations) in transitions.iter() {
            let prefix_str: String = prefix.extract()?;
            let cont_list: Vec<String> = continuations.extract()?;
            paradigm
                .allowed_transitions
                .insert(prefix_str, cont_list.into_iter().collect());
        }
        self.inner.paradigm_registry.add_paradigm(paradigm);
        Ok(())
    }

    fn train_bpe(
        &mut self,
        sequences: Vec<Vec<usize>>,
        vocab_budget: usize,
        theta: u64,
    ) -> PyResult<usize> {
        // Wire paradigm roots into the vocab before training. No-op behaviorally
        // if no paradigms have been registered.
        self.inner
            .vocab
            .assign_roots_from_registry(&self.inner.paradigm_registry);

        let mut corpus = Corpus::from_sequences(sequences, vocab_budget);
        let mut trainer = ConstrainedBPETrainer::new(
            std::mem::take(&mut self.inner.vocab),
            std::mem::take(&mut self.inner.paradigm_registry),
        );
        trainer.theta = theta;
        trainer.train(&mut corpus, vocab_budget, 0);

        self.inner.vocab = trainer.vocab;
        self.inner.paradigm_registry = trainer.paradigm_registry;

        Ok(self.inner.vocab.len())
    }

    /// Streaming, word-deduplicated training over a text file — the path to use
    /// for large corpora (multi-GB). Reads the file line by line, so peak RAM is
    /// the word-frequency dictionary, NOT the tokenized corpus.
    ///
    /// - `min_word_freq`: drop word types occurring fewer than this many times
    ///   (cuts the huge hapax tail of morphologically rich Nepali; 1 = keep all).
    /// - `progress_lines`: print a build-progress line every N input lines (0=off).
    /// - `progress_merges`: print a train-progress line every N merges (0=off).
    ///
    /// Timing for the build phase and the train phase is printed to stderr. The
    /// return value is the final vocab size.
    #[pyo3(signature = (path, vocab_budget, theta, min_word_freq=1, progress_lines=500000, progress_merges=1000))]
    fn train_from_file(
        &mut self,
        py: Python<'_>,
        path: String,
        vocab_budget: usize,
        theta: u64,
        min_word_freq: u64,
        progress_lines: u64,
        progress_merges: u64,
    ) -> PyResult<usize> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};
        use std::time::Instant;

        let file = File::open(&path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("open {}: {}", path, e))
        })?;

        // Release the GIL for the whole heavy phase; we only touch pure-Rust state.
        let (counts, lines_done, word_occ, build_secs) = py.allow_threads(|| {
            let t0 = Instant::now();
            let reader = BufReader::with_capacity(1 << 20, file);
            let mut counts: HashMap<Vec<TokenId>, Frequency> = HashMap::new();
            let mut lines_done: u64 = 0;
            let mut word_occ: u64 = 0;

            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => continue, // skip unreadable line rather than abort a long job
                };
                let norm = self.inner.normalizer.normalize(&line);
                for word in norm.split_whitespace() {
                    let toks = self.inner.encode_normalized(word);
                    if !toks.is_empty() {
                        *counts.entry(toks).or_insert(0) += 1;
                        word_occ += 1;
                    }
                }
                lines_done += 1;
                if progress_lines > 0 && lines_done % progress_lines == 0 {
                    eprintln!(
                        "[build] {} lines | {} word-occ | {} unique | {:.1}s",
                        lines_done,
                        word_occ,
                        counts.len(),
                        t0.elapsed().as_secs_f64()
                    );
                }
            }

            if min_word_freq > 1 {
                counts.retain(|_, &mut c| c >= min_word_freq);
            }
            (counts, lines_done, word_occ, t0.elapsed().as_secs_f64())
        });

        eprintln!(
            "[build] done: {} lines, {} word-occ, {} unique types kept (min_freq={}) in {:.1}s",
            lines_done,
            word_occ,
            counts.len(),
            min_word_freq,
            build_secs
        );

        // Tag roots (no-op unless paradigms loaded), then train (GIL released).
        self.inner
            .vocab
            .assign_roots_from_registry(&self.inner.paradigm_registry);

        let final_vocab = py.allow_threads(|| {
            let mut corpus = Corpus::from_word_counts(counts, vocab_budget);
            let mut trainer = ConstrainedBPETrainer::new(
                std::mem::take(&mut self.inner.vocab),
                std::mem::take(&mut self.inner.paradigm_registry),
            );
            trainer.theta = theta;
            trainer.train(&mut corpus, vocab_budget, progress_merges);

            self.inner.vocab = trainer.vocab;
            self.inner.paradigm_registry = trainer.paradigm_registry;
            self.inner.vocab.len()
        });

        Ok(final_vocab)
    }

    fn train_from_text(
        &mut self,
        texts: Vec<String>,
        vocab_budget: usize,
        theta: u64,
    ) -> PyResult<usize> {
        let sequences: Vec<Vec<usize>> = texts
            .iter()
            .map(|t| self.inner.encode(t))
            .filter(|seq| !seq.is_empty())
            .collect();

        if sequences.is_empty() {
            return Ok(self.inner.vocab.len());
        }

        self.train_bpe(sequences, vocab_budget, theta)
    }

    /// PHASE 4 — bilingual training. Builds the word-frequency dictionary ONCE
    /// from a mixed Nepali+English file, then runs two passes over it:
    ///
    ///   1. Phase 3 (constrained): Devanagari + punctuation, up to `dev_budget`.
    ///   2. Phase 4 (unconstrained): Latin/digits only, up to
    ///      `dev_budget + lat_budget` total.
    ///
    /// The budget split is an explicit CHOICE, not a default. Nepali is the
    /// priority language and needs the larger slice; English reaches acceptable
    /// fertility with far fewer slots (its high-frequency subword core is small).
    /// A reasonable start is dev=40000, lat=8000.
    ///
    /// Script separation is preserved: no DEV/LAT token can ever form, because
    /// each pass's ScriptCompat admits only its own script.
    #[pyo3(signature = (path, dev_budget, lat_budget, theta, min_word_freq=1, progress_lines=500000, progress_merges=1000))]
    fn train_bilingual_from_file(
        &mut self,
        py: Python<'_>,
        path: String,
        dev_budget: usize,
        lat_budget: usize,
        theta: u64,
        min_word_freq: u64,
        progress_lines: u64,
        progress_merges: u64,
    ) -> PyResult<usize> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};
        use std::time::Instant;

        let file = File::open(&path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("open {}: {}", path, e))
        })?;

        let (counts, lines_done, word_occ, build_secs) = py.allow_threads(|| {
            let t0 = Instant::now();
            let reader = BufReader::with_capacity(1 << 20, file);
            let mut counts: HashMap<Vec<TokenId>, Frequency> = HashMap::new();
            let mut lines_done: u64 = 0;
            let mut word_occ: u64 = 0;

            for line in reader.lines() {
                let line = match line {
                    Ok(l) => l,
                    Err(_) => continue,
                };
                let norm = self.inner.normalizer.normalize(&line);
                for word in norm.split_whitespace() {
                    let toks = self.inner.encode_normalized(word);
                    if !toks.is_empty() {
                        *counts.entry(toks).or_insert(0) += 1;
                        word_occ += 1;
                    }
                }
                lines_done += 1;
                if progress_lines > 0 && lines_done % progress_lines == 0 {
                    eprintln!(
                        "[build] {} lines | {} word-occ | {} unique | {:.1}s",
                        lines_done,
                        word_occ,
                        counts.len(),
                        t0.elapsed().as_secs_f64()
                    );
                }
            }
            if min_word_freq > 1 {
                counts.retain(|_, &mut c| c >= min_word_freq);
            }
            (counts, lines_done, word_occ, t0.elapsed().as_secs_f64())
        });

        eprintln!(
            "[build] done: {} lines, {} word-occ, {} unique types kept (min_freq={}) in {:.1}s",
            lines_done, word_occ, counts.len(), min_word_freq, build_secs
        );

        self.inner
            .vocab
            .assign_roots_from_registry(&self.inner.paradigm_registry);

        let total_budget = dev_budget + lat_budget;

        let final_vocab = py.allow_threads(|| {
            let mut corpus = Corpus::from_word_counts(counts, total_budget);
            let mut trainer = ConstrainedBPETrainer::new(
                std::mem::take(&mut self.inner.vocab),
                std::mem::take(&mut self.inner.paradigm_registry),
            );
            trainer.theta = theta;

            // Pass 1 — Phase 3, constrained, Devanagari + punctuation.
            trainer.latin_pass = false;
            trainer.train(&mut corpus, dev_budget, progress_merges);
            let after_dev = trainer.vocab.len();

            // Pass 2 — Phase 4, unconstrained, Latin only. The heap is rebuilt
            // from scratch inside train(), and initialize_heap now admits LAT
            // pairs because script_compat flipped.
            trainer.latin_pass = true;
            trainer.train(&mut corpus, total_budget, progress_merges);
            let after_lat = trainer.vocab.len();

            eprintln!(
                "[train] budget split: DEV {} (target {}) | LAT +{} (target +{})",
                after_dev,
                dev_budget,
                after_lat - after_dev,
                lat_budget
            );

            self.inner.vocab = trainer.vocab;
            self.inner.paradigm_registry = trainer.paradigm_registry;
            self.inner.vocab.len()
        });

        Ok(final_vocab)
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab.len()
    }

    /// Rebuild the vocab from (id, surface) pairs — encode/decode-ready, not
    /// training-ready. Ids must be contiguous 0..N.
    fn load_vocab(&mut self, pairs: Vec<(usize, String)>) -> PyResult<usize> {
        self.inner.load_vocab(pairs);
        Ok(self.inner.vocab.len())
    }

    /// Load the vocab from a TSV written by the training driver ("id\tsurface"),
    /// reversing the tab/newline/backslash escaping. Encode/decode-ready.
    fn load_vocab_tsv(&mut self, path: String) -> PyResult<usize> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(&path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("open {}: {}", path, e))
        })?;
        let reader = BufReader::new(file);

        let mut pairs: Vec<(usize, String)> = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("read: {}", e))
            })?;
            if line.is_empty() {
                continue;
            }
            let mut it = line.splitn(2, '\t');
            let id_str = match it.next() {
                Some(s) => s,
                None => continue,
            };
            let surf_raw = it.next().unwrap_or("");
            let id: usize = match id_str.parse() {
                Ok(v) => v,
                Err(_) => continue, // skip a malformed line rather than abort
            };
            let surface = unescape_tsv(surf_raw);
            pairs.push((id, surface));
        }

        self.inner.load_vocab(pairs);
        Ok(self.inner.vocab.len())
    }

    fn get_token_surface(&self, id: usize) -> PyResult<String> {
        self.inner
            .vocab
            .get_surface(id)
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid token ID"))
    }
}

#[pymodule]
fn tiny_llm_scratch_with_tokenizer(m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNepBPETokenizer>()?;
    Ok(())
}