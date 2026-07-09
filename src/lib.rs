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
// Phase 1: Normalization N
// ============================================================================

pub struct Normalizer {
    folding_table: HashMap<char, char>,
}

impl Normalizer {
    pub fn new(folding_table: HashMap<char, char>) -> Self {
        Self { folding_table }
    }

    pub fn normalize(&self, s: &str) -> String {
        let nfc: String = s.nfc().collect();
        let folded: String = nfc
            .chars()
            .map(|c| self.folding_table.get(&c).copied().unwrap_or(c))
            .collect();
        folded.chars().filter(|&c| c != '\u{200D}').collect()
    }
}

// ============================================================================
// Phase 2: Akshara DFA - FIXED VERSION
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
            let mut last_accepting_state: Option<usize> = None;
            let mut j = i;

            // Try to find the longest accepting path
            while j < chars.len() {
                let ch = chars[j];
                if let Some(&next_state) = self.transitions.get(&(state, ch)) {
                    state = next_state;
                    if self.accepting_states.contains(&state) {
                        last_accepting_pos = Some(j + 1);
                        last_accepting_state = Some(state);
                    }
                    j += 1;
                } else {
                    // No transition - stop here
                    break;
                }
            }

            if let Some(end) = last_accepting_pos {
                // Found a valid akshara
                let akshara_str: String = chars[i..end].iter().collect();
                let root_set = Vec::new(); // Will be computed by paradigm registry
                aksharas.push(Akshara {
                    surface: akshara_str,
                    root_set,
                });
                i = end;
            } else {
                // No valid akshara found - emit single character as fallback
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
}

impl Token {
    pub fn script(&self) -> Script {
        match self {
            Token::Akshara(_) => Script::DEV,
            Token::Punctuation(_) => Script::PUN,
            Token::ZWNJ => Script::FMT,
            Token::ByteFallback(_) => Script::MAL,
            Token::SeededMorpheme(_) => Script::DEV,
            Token::MergedToken(_) => Script::DEV,
        }
    }

    pub fn base_surface(&self) -> String {
        match self {
            Token::Akshara(a) => a.surface.clone(),
            Token::Punctuation(p) => p.clone(),
            Token::ZWNJ => '\u{200C}'.to_string(),
            Token::ByteFallback(b) => (*b as char).to_string(),
            Token::SeededMorpheme(s) => s.clone(),
            Token::MergedToken(_) => String::new(),
        }
    }
}

// ============================================================================
// Vocabulary management - FIXED: Use surface string as primary key
// ============================================================================

#[derive(Default)]
pub struct Vocabulary {
    tokens: Vec<Arc<Token>>,
    /// CRITICAL FIX: Use surface string as primary lookup key
    surface_to_id: HashMap<String, TokenId>,
    id_to_script: HashMap<TokenId, Script>,
    v_strict: HashSet<TokenId>,
    v_ambiguous: HashSet<TokenId>,
    token_to_root_set: HashMap<TokenId, Vec<RootId>>,
    surfaces: HashMap<TokenId, String>,
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
    ) {
        for akshara in base_aksharas {
            let surface = akshara.surface.clone();
            let token = Arc::new(Token::Akshara(akshara));
            let id = self.add_token(token.clone(), surface.clone(), false, false);
        }
        for morph in seed_morphemes {
            let surface = morph.clone();
            let is_strict = v_strict.contains(&morph);
            let is_ambiguous = v_ambiguous.contains(&morph);
            let token = Arc::new(Token::SeededMorpheme(morph));
            let id = self.add_token(token.clone(), surface.clone(), is_strict, is_ambiguous);
        }
        for punct in punctuation {
            let surface = punct.clone();
            let token = Arc::new(Token::Punctuation(punct));
            let id = self.add_token(token.clone(), surface.clone(), false, false);
        }
        for byte_val in 0u8..=255 {
            let surface = format!("<0x{:02X}>", byte_val);
            let token = Arc::new(Token::ByteFallback(byte_val));
            let id = self.add_token(token.clone(), surface.clone(), false, false);
        }
        let zwnj = Arc::new(Token::ZWNJ);
        let id = self.add_token(zwnj.clone(), "\u{200C}".to_string(), false, false);
    }

    /// FIXED: Add token with surface string as primary key
    fn add_token(
        &mut self,
        token: Arc<Token>,
        surface: String,
        is_strict: bool,
        is_ambiguous: bool,
    ) -> TokenId {
        // Check if surface already exists
        if let Some(&existing_id) = self.surface_to_id.get(&surface) {
            return existing_id;
        }

        let id = self.tokens.len();
        let script = token.script();
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

    /// FIXED: Lookup by surface string instead of Arc<Token>
    pub fn get_id_by_surface(&self, surface: &str) -> Option<TokenId> {
        self.surface_to_id.get(surface).copied()
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

    pub fn create_merged(&mut self, a: TokenId, b: TokenId, merged_root_set: Vec<RootId>) -> TokenId {
        let surface_a = self.get_surface(a).unwrap_or_default();
        let surface_b = self.get_surface(b).unwrap_or_default();
        let merged_surface = format!("{}{}", surface_a, surface_b);

        let merged = Arc::new(Token::MergedToken(vec![a, b]));
        let id = self.add_token(merged, merged_surface, false, false);
        self.token_to_root_set.insert(id, merged_root_set);
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

pub struct Corpus {
    words: Vec<Vec<TokenId>>,
    pair_freqs: HashMap<(TokenId, TokenId), Frequency>,
    vocab_budget: usize,
}

impl Corpus {
    pub fn new(sequences: Vec<Vec<TokenId>>, vocab_budget: usize) -> Self {
        let mut corpus = Self {
            words: sequences,
            pair_freqs: HashMap::new(),
            vocab_budget,
        };
        corpus.recompute_all_frequencies();
        corpus
    }

    fn recompute_all_frequencies(&mut self) {
        self.pair_freqs.clear();
        for word in &self.words {
            for window in word.windows(2) {
                *self.pair_freqs
                    .entry((window[0], window[1]))
                    .or_insert(0) += 1;
            }
        }
    }

    pub fn get_freq(&self, a: TokenId, b: TokenId) -> Frequency {
        self.pair_freqs.get(&(a, b)).copied().unwrap_or(0)
    }

    pub fn apply_merge(&mut self, a: TokenId, b: TokenId, new_id: TokenId) {
        for word in &mut self.words {
            let mut i = 0;
            while i + 1 < word.len() {
                if word[i] == a && word[i + 1] == b {
                    word[i] = new_id;
                    word.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }
        self.recompute_all_frequencies();
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
        self.priority_key == other.priority_key
            && self.a == other.a
            && self.b == other.b
    }
}
impl Eq for MergeCandidate {}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority_key.cmp(&other.priority_key).reverse()
    }
}

pub struct ConstrainedBPETrainer {
    vocab: Vocabulary,
    paradigm_registry: ParadigmRegistry,
    pub theta: Frequency,
}

impl ConstrainedBPETrainer {
    pub fn new(vocab: Vocabulary, paradigm_registry: ParadigmRegistry) -> Self {
        Self {
            vocab,
            paradigm_registry,
            theta: 100,
        }
    }

    fn script_compat(&self, a: TokenId, b: TokenId) -> bool {
        let sa = self.vocab.get_script(a);
        let sb = self.vocab.get_script(b);
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
        let root_set = self.vocab.get_root_set(a);
        if root_set.is_empty() {
            return true;
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
        (self.vocab.get_script(a).rank(), freq as u64)
    }

    pub fn train(&mut self, corpus: &mut Corpus) {
        let mut heap: BinaryHeap<MergeCandidate> = BinaryHeap::new();
        self.initialize_heap(corpus, &mut heap);

        while self.vocab.len() < corpus.vocab_budget {
            let mut found_valid = false;

            while let Some(candidate) = heap.pop() {
                let current_freq = corpus.get_freq(candidate.a, candidate.b);

                if current_freq != candidate.freq_snapshot {
                    continue;
                }

                if !self.legal(candidate.a, candidate.b, current_freq) {
                    continue;
                }

                let narrowed_roots = self.narrow_root_set(candidate.a, candidate.b);
                let new_id = self
                    .vocab
                    .create_merged(candidate.a, candidate.b, narrowed_roots);

                corpus.apply_merge(candidate.a, candidate.b, new_id);
                self.push_affected_pairs(corpus, &mut heap);
                found_valid = true;
                break;
            }

            if !found_valid {
                break;
            }
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

    fn push_affected_pairs(
        &self,
        corpus: &Corpus,
        heap: &mut BinaryHeap<MergeCandidate>,
    ) {
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
// Main tokenizer: FIXED VERSION
// ============================================================================

#[allow(dead_code)]
pub struct NepBPETokenizer {
    pub normalizer: Normalizer,
    pub akshara_dfa: AksharaDFA,
    pub vocab: Vocabulary,
    pub paradigm_registry: ParadigmRegistry,
    trainer: Option<ConstrainedBPETrainer>,
    paradigm_embedding: HashMap<TokenId, Option<RootId>>,
}

impl NepBPETokenizer {
    pub fn new(folding_table: HashMap<char, char>, paradigm_registry: ParadigmRegistry) -> Self {
        Self {
            normalizer: Normalizer::new(folding_table),
            akshara_dfa: AksharaDFA::new(),
            vocab: Vocabulary::new(),
            paradigm_registry,
            trainer: None,
            paradigm_embedding: HashMap::new(),
        }
    }

    pub fn encode(&self, s: &str) -> Vec<TokenId> {
        let normalized = self.normalizer.normalize(s);
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

       /// FIXED: Use surface string lookup instead of Arc<Token> lookup
    fn tokenize_run(&self, run: &str, script: Script, tokens: &mut Vec<TokenId>) {
        match script {
            Script::DEV => {
                let aksharas = self.akshara_dfa.tokenize(run);
                for akshara in aksharas {
                    if let Some(id) = self.vocab.get_id_by_surface(&akshara.surface) {
                        tokens.push(id);
                    } else {
                        self.fallback_tokenize_dev(&akshara.surface, tokens);
                    }
                }
            }
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
            Script::LAT => {
                // FIX: Try longest vocabulary match first (e.g., "hello", "nepal")
                let chars: Vec<char> = run.chars().collect();
                let mut i = 0;
                while i < chars.len() {
                    let mut found = false;
                    for len in (1..=chars.len() - i).rev() {
                        let candidate: String = chars[i..i+len].iter().collect();
                        if let Some(id) = self.vocab.get_id_by_surface(&candidate) {
                            tokens.push(id);
                            i += len;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        let ch_str = chars[i].to_string();
                        self.emit_byte_fallback(ch_str.as_bytes(), tokens);
                        i += 1;
                    }
                }
            }
        }
    }

    fn emit_byte_fallback(&self, bytes: &[u8], tokens: &mut Vec<TokenId>) {
        for &byte in bytes {
            let surface = format!("<0x{:02X}>", byte);
            if let Some(id) = self.vocab.get_id_by_surface(&surface) {
                tokens.push(id);
            }
        }
    }

    fn fallback_tokenize_dev(&self, text: &str, tokens: &mut Vec<TokenId>) {
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        
        while i < chars.len() {
            let mut found = false;
            for len in (2..=chars.len() - i).rev() {
                let candidate: String = chars[i..i+len].iter().collect();
                if let Some(id) = self.vocab.get_id_by_surface(&candidate) {
                    tokens.push(id);
                    i += len;
                    found = true;
                    break;
                }
            }
            if !found {
                let ch = chars[i];
                if let Some(id) = self.vocab.get_id_by_surface(&ch.to_string()) {
                    tokens.push(id);
                } else {
                    for byte in ch.to_string().as_bytes() {
                        let surface = format!("<0x{:02X}>", byte);
                        if let Some(id) = self.vocab.get_id_by_surface(&surface) {
                            tokens.push(id);
                        }
                    }
                }
                i += 1;
            }
        }
    }

    // FIX: Reconstruct UTF-8 from byte fallback tokens instead of deleting them
    pub fn decode(&self, token_ids: &[TokenId]) -> String {
        let mut result = String::new();
        let mut byte_buf: Vec<u8> = Vec::new();
        
        for &id in token_ids {
            if let Some(surface) = self.vocab.get_surface(id) {
                // Check if this is a byte fallback token <0xXX>
                if surface.len() == 6 && surface.starts_with("<0x") && surface.ends_with('>') {
                    if let Ok(byte_val) = u8::from_str_radix(&surface[3..5], 16) {
                        byte_buf.push(byte_val);
                        continue;
                    }
                }
                
                // Not a byte fallback - flush any pending bytes first
                if !byte_buf.is_empty() {
                    match String::from_utf8(std::mem::take(&mut byte_buf)) {
                        Ok(s) => result.push_str(&s),
                        Err(_) => {} // Invalid UTF-8 sequence, skip
                    }
                }
                
                // Add the surface to result (skip ZWNJ)
                if surface != "\u{200C}" {
                    result.push_str(&surface);
                }
            }
        }
        
        // Flush any remaining bytes at the very end
        if !byte_buf.is_empty() {
            if let Ok(s) = String::from_utf8(byte_buf) {
                result.push_str(&s);
            }
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
// Python bindings - FIXED with debug methods
// ============================================================================

#[pyclass]
pub struct PyNepBPETokenizer {
    inner: NepBPETokenizer,
}

#[pymethods]
impl PyNepBPETokenizer {
    #[new]
    #[pyo3(signature = (folding_table=None))]
    fn new<'py>(folding_table: Option<Bound<'py, PyDict>>) -> PyResult<Self> {
        let mut fold = HashMap::new();
        if let Some(py_dict) = folding_table {
            for (k, v) in py_dict.iter() {
                let key: char = k.extract()?;
                let val: char = v.extract()?;
                fold.insert(key, val);
            }
        }
        let registry = ParadigmRegistry::new();
        let inner = NepBPETokenizer::new(fold, registry);
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
        );

        Ok(self.inner.vocab.len())
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
        let mut corpus = Corpus::new(sequences, vocab_budget);
        let mut trainer = ConstrainedBPETrainer::new(
            std::mem::take(&mut self.inner.vocab),
            std::mem::take(&mut self.inner.paradigm_registry),
        );
        trainer.theta = theta;
        trainer.train(&mut corpus);

        self.inner.vocab = trainer.vocab;
        self.inner.paradigm_registry = trainer.paradigm_registry;

        Ok(self.inner.vocab.len())
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

    fn vocab_size(&self) -> usize {
        self.inner.vocab.len()
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