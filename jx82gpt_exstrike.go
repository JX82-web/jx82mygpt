// jx82gpt - assistente da terminale per studio e lavoro quotidiano.
// Il programma propone, spiega e prepara. L'utente decide sempre cosa eseguire e cosa salvare.
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// ============================================================================
// Costanti e variabili globali
// ============================================================================

const (
	maxOutputRunes     = 20000
	maxFileBytes       = 1 << 20
	maxPipeBytes       = 10 << 20
	defaultOllamaModel = "qwen3-coder:30b"
)

var (
	reset  = "\033[0m"
	bold   = "\033[1m"
	red    = "\033[31m"
	green  = "\033[32m"
	yellow = "\033[33m"
	cyan   = "\033[36m"
	gray   = "\033[90m"
)

// initColors spegne i colori se NO_COLOR è impostata o se l'output non è un terminale.
func initColors() {
	stat, err := os.Stdout.Stat()
	isTerminal := err == nil && (stat.Mode()&os.ModeCharDevice) != 0
	if os.Getenv("NO_COLOR") != "" || !isTerminal {
		reset, bold, red, green, yellow, cyan, gray = "", "", "", "", "", "", ""
	}
}

var errUsage = errors.New("specifica una domanda, oppure usa -i per la modalità interattiva (-h per la guida)")

var providerDefaultModel = map[string]string{
	"anthropic": "claude-sonnet-5-5",
	"openai":    "gpt-4o",
}

var apiKeyEnv = map[string]string{
	"anthropic": "ANTHROPIC_API_KEY",
	"openai":    "OPENAI_API_KEY",
}

var httpClient = &http.Client{}

// hexStrikeClient parla con il backend HTTP di HexStrike.
// Quando -s è impostato, i blocchi shell approvati vengono eseguiti
// dal server HexStrike invece che dalla shell locale.
type hexStrikeClient struct {
	baseURL string
	timeout time.Duration
}

func newHexStrikeClient(rawURL string, timeout time.Duration) (*hexStrikeClient, error) {
	rawURL = strings.TrimSpace(rawURL)
	if rawURL == "" {
		return nil, errors.New("indirizzo HexStrike vuoto")
	}
	if !strings.Contains(rawURL, "://") {
		rawURL = "http://" + rawURL
	}
	rawURL = strings.TrimRight(rawURL, "/")

	u, err := url.Parse(rawURL)
	if err != nil || u.Scheme == "" || u.Host == "" {
		return nil, fmt.Errorf("indirizzo HexStrike non valido: %q", rawURL)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return nil, fmt.Errorf("schema HexStrike non supportato: %q (usa http o https)", u.Scheme)
	}
	return &hexStrikeClient{baseURL: rawURL, timeout: timeout}, nil
}

func (h *hexStrikeClient) health(ctx context.Context) (map[string]any, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, h.baseURL+"/health", nil)
	if err != nil {
		return nil, err
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HexStrike non raggiungibile su %s: %w", h.baseURL, err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("HexStrike /health ha risposto %d: %s", resp.StatusCode, clip(string(raw), 500))
	}

	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("risposta /health di HexStrike non valida: %w", err)
	}
	return out, nil
}

func (h *hexStrikeClient) command(ctx context.Context, command string) (string, int, error) {
	payload, err := json.Marshal(map[string]any{
		"command":   command,
		"use_cache": true,
	})
	if err != nil {
		return "", -1, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.baseURL+"/api/command", bytes.NewReader(payload))
	if err != nil {
		return "", -1, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return "", -1, fmt.Errorf("richiesta HexStrike fallita: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return "", -1, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", -1, fmt.Errorf("HexStrike /api/command ha risposto %d: %s", resp.StatusCode, clip(string(raw), 1000))
	}

	// L'API di HexStrike può restituire campi diversi tra versioni.
	// Prima proviamo le forme più comuni, poi mostriamo il JSON completo.
	var out struct {
		Output  string `json:"output"`
		Stdout  string `json:"stdout"`
		Result  string `json:"result"`
		Code    *int   `json:"return_code"`
		Status  *int   `json:"exit_code"`
		Error   string `json:"error"`
		Success *bool  `json:"success"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return string(raw), 0, nil
	}
	if out.Error != "" {
		return "", -1, errors.New("HexStrike: " + out.Error)
	}

	text := out.Output
	if text == "" {
		text = out.Stdout
	}
	if text == "" {
		text = out.Result
	}
	if text == "" {
		text = string(raw)
	}

	code := 0
	if out.Code != nil {
		code = *out.Code
	} else if out.Status != nil {
		code = *out.Status
	}
	if out.Success != nil && !*out.Success && code == 0 {
		code = 1
	}
	return text, code, nil
}

var (
	fileBlockRe = regexp.MustCompile("(?ms)^FILE:[ \\t]*(\\S+)[ \\t]*\\r?\\n```[^\\n]*\\n(.*?)^```[ \\t]*$")
	codeBlockRe = regexp.MustCompile("(?ms)^```([A-Za-z0-9_+.-]*)[ \\t]*\\r?\\n(.*?)^```[ \\t]*$")
)

var shellLangs = map[string]bool{"bash": true, "sh": true, "zsh": true}

// Prompt di sistema predefinito. Puoi sostituirlo con ~/.jx82mygpt/prompt.txt.
// const defaultPrompt = `Sei un assistente tecnico per studio e lavoro quotidiano. Rispondi sempre in italiano.
const defaultPrompt = `Sei un esperto senior di cybersecurity e programmazione generale >

Come lavori:
- Parti dai fondamenti meccanici: come funziona il componente, quali assunzioni di fiduc>
- Per ogni vulnerabilità descrivi cause, condizioni necessarie, impatto, come riconoscer>
- Analizza CVE, advisory, report di incidenti e pubblicazioni di ricerca: contesto, vers>
- Usa i framework di riferimento in modo preciso: MITRE ATT&CK e D3FEND, OWASP Top 10 e >
- Per la preparazione agli esami, costruisci piani di studio con priorità, domande di ve>
- Quando lo studente condivide codice, configurazioni o architetture, fai una revisione >
- Indica risorse gratuite e legali: documentazione ufficiale, PortSwigger Web Security A>
- Non ripeti mai a lungo concetti o vulnerabilità trovate ma esequi esempi pratici di at>
- Non menti mai e resti privato ed in locale come un vero senior del settore RedTeam.
Come aiuti:
- Spiega i concetti dai fondamenti: protocolli, sistemi operativi, reti, crittografia, g>
- Per ogni vulnerabilità spiega cause, impatto, come riconoscerla leggendo codice o conf>
- Aiuta a leggere CVE, advisory, report di incidenti e framework come MITRE ATT&CK, OWAS>
- Prepara lo studente agli esami: crea piani di studio, quiz, domande di ripasso e spieg>
- Indica risorse gratuite e legali: PortSwigger Web Security Academy, TryHackMe, OWASP, >
- Quando lo studente analizza il proprio codice o le proprie configurazioni, cerca i pro>
Come rispondi:
- Spiega ogni passaggio in modo chiaro e verificabile, dicendo perché un comando o una s>
- Quando proponi un comando da eseguire, mettilo in un blocco di codice bash commentato >
- Non simulare mai output di terminale, log o risultati. Indica cosa aspettarti e lascia>
- Se ti chiedono di creare file o un progetto, produci ogni file così: una riga "FILE: p>
- Per la sicurezza informatica concentrati su spiegazione, difesa, configurazione sicura>
- Se una richiesta è ambigua, chiedi un chiarimento prima di produrre file o comandi.
`
// ============================================================================
// Tipi e configurazione
// ============================================================================

// Config contiene le impostazioni. I campi con json:"-" arrivano solo dalla riga di comando.
type Config struct {
	Provider        string  `json:"provider"`
	Model           string  `json:"model"`
	OllamaURL       string  `json:"ollama_url"`
	Temperature     float64 `json:"temperature"`
	TimeoutSeconds  int     `json:"timeout_seconds"`
	ShellTimeoutSec int     `json:"shell_timeout_seconds"`
	MaxMessages     int     `json:"max_messages"`
	SpendCapTokens  int     `json:"spend_cap_tokens"`
	WorkRoot        string  `json:"work_root"`
	LogPath         string  `json:"log_path"`
	HexStrikeURL    string  `json:"-"`

	Cloud       bool   `json:"-"`
	DryRun      bool   `json:"-"`
	Interactive bool   `json:"-"`
	Project     string `json:"-"`
	ConfigDir   string `json:"-"`
}

func defaultConfig() *Config {
	home, _ := os.UserHomeDir()
	base := filepath.Join(home, ".jx82mygpt")
	return &Config{
		Provider:        "ollama",
		Model:           defaultOllamaModel,
		OllamaURL:       "http://127.0.0.1:11434",
		Temperature:     0.1,
		TimeoutSeconds:  600,
		ShellTimeoutSec: 300,
		MaxMessages:     40,
		WorkRoot:        filepath.Join(home, "jx82-progetti"),
		LogPath:         filepath.Join(base, "log.jsonl"),
		ConfigDir:       base,
	}
}

func (c *Config) Timeout() time.Duration      { return time.Duration(c.TimeoutSeconds) * time.Second }
func (c *Config) ShellTimeout() time.Duration { return time.Duration(c.ShellTimeoutSec) * time.Second }

// loadConfigFile applica config.json se esiste. Variabile JX82_CONFIG per un percorso alternativo.
func loadConfigFile(cfg *Config) error {
	path := os.Getenv("JX82_CONFIG")
	if path == "" {
		path = filepath.Join(cfg.ConfigDir, "config.json")
	}
	raw, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	if err := json.Unmarshal(raw, cfg); err != nil {
		return fmt.Errorf("config %s non valido: %w", path, err)
	}
	return nil
}

// resolve rende coerenti le opzioni: modalità cloud o locale, modello di default per provider.
func (c *Config) resolve() error {
	if c.Cloud {
		if c.Provider == "" || c.Provider == "ollama" {
			c.Provider = "anthropic"
		}
		def, ok := providerDefaultModel[c.Provider]
		if !ok {
			return fmt.Errorf("provider cloud non supportato: %q (usa anthropic o openai)", c.Provider)
		}
		if c.Model == defaultOllamaModel {
			c.Model = def
		}
	} else {
		c.Provider = "ollama"
	}
	if c.Model == "" {
		return errors.New("modello non specificato")
	}
	return nil
}

// loadPrompt usa ~/.jx82mygpt/prompt.txt se presente, altrimenti il prompt predefinito.
func loadPrompt(cfg *Config) string {
	raw, err := os.ReadFile(filepath.Join(cfg.ConfigDir, "prompt.txt"))
	if err == nil && strings.TrimSpace(string(raw)) != "" {
		return string(raw)
	}
	return defaultPrompt
}

// ============================================================================
// Client dei modelli: Ollama (locale), Anthropic e OpenAI (cloud)
// ============================================================================

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Reply struct {
	Text      string
	InTokens  int
	OutTokens int
}

// Client è l'interfaccia comune a tutti i provider.
type Client interface {
	Chat(ctx context.Context, msgs []Message, onToken func(string)) (Reply, error)
}

// newClient restituisce il client e l'elenco dei segreti da non stampare mai.
func newClient(cfg *Config) (Client, []string, error) {
	switch cfg.Provider {
	case "ollama":
		return &ollamaClient{url: strings.TrimRight(cfg.OllamaURL, "/"), model: cfg.Model, temp: cfg.Temperature}, nil, nil
	case "anthropic", "openai":
		env := apiKeyEnv[cfg.Provider]
		key := os.Getenv(env)
		if key == "" {
			return nil, nil, fmt.Errorf("variabile %s non impostata: il provider %s richiede una chiave API", env, cfg.Provider)
		}
		if cfg.Provider == "anthropic" {
			return &anthropicClient{key: key, model: cfg.Model}, []string{key}, nil
		}
		return &openaiClient{key: key, model: cfg.Model}, []string{key}, nil
	}
	return nil, nil, fmt.Errorf("provider sconosciuto: %s", cfg.Provider)
}

func redact(s string, secrets []string) string {
	for _, k := range secrets {
		if k != "" {
			s = strings.ReplaceAll(s, k, "***")
		}
	}
	return s
}

// ---------- Ollama (locale, streaming) ----------

type ollamaClient struct {
	url, model string
	temp       float64
}

func (c *ollamaClient) Chat(ctx context.Context, msgs []Message, onToken func(string)) (Reply, error) {
	body, _ := json.Marshal(map[string]any{
		"model":    c.model,
		"messages": msgs,
		"stream":   true,
		"options": map[string]any{
			"temperature": c.temp,
			"num_ctx":     8192,
		},
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return Reply{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return Reply{}, fmt.Errorf("Ollama non raggiungibile su %s: avvialo con 'ollama serve' (%v)", c.url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return Reply{}, fmt.Errorf("modello %q non trovato: installalo con 'ollama pull %s'", c.model, c.model)
	}
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return Reply{}, fmt.Errorf("Ollama ha risposto %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 16*1024*1024)
	var full strings.Builder
	var rep Reply
	for scanner.Scan() {
		var chunk struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			Done            bool   `json:"done"`
			Error           string `json:"error"`
			PromptEvalCount int    `json:"prompt_eval_count"`
			EvalCount       int    `json:"eval_count"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &chunk); err != nil {
			continue
		}
		if chunk.Error != "" {
			return rep, errors.New("Ollama: " + chunk.Error)
		}
		if chunk.Message.Content != "" {
			full.WriteString(chunk.Message.Content)
			onToken(chunk.Message.Content)
		}
		if chunk.Done {
			rep.InTokens = chunk.PromptEvalCount
			rep.OutTokens = chunk.EvalCount
		}
	}
	if err := scanner.Err(); err != nil {
		return rep, err
	}
	rep.Text = full.String()
	return rep, nil
}

// ---------- Anthropic (cloud, risposta completa) ----------

type anthropicClient struct {
	key, model string
}

func (c *anthropicClient) Chat(ctx context.Context, msgs []Message, onToken func(string)) (Reply, error) {
	var system []string
	conv := make([]Message, 0, len(msgs))
	for _, m := range msgs {
		if m.Role == "system" {
			system = append(system, m.Content)
			continue
		}
		conv = append(conv, m)
	}
	payload := map[string]any{
		"model":      c.model,
		"max_tokens": 4096,
		"messages":   conv,
	}
	if len(system) > 0 {
		payload["system"] = strings.Join(system, "\n")
	}
	body, _ := json.Marshal(payload)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.anthropic.com/v1/messages", bytes.NewReader(body))
	if err != nil {
		return Reply{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.key)
	req.Header.Set("anthropic-version", "2023-06-01")

	var out struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := doJSON(req, &out, []string{c.key}); err != nil {
		return Reply{}, err
	}
	var sb strings.Builder
	for _, part := range out.Content {
		if part.Type == "text" {
			sb.WriteString(part.Text)
		}
	}
	text := sb.String()
	onToken(text)
	return Reply{Text: text, InTokens: out.Usage.InputTokens, OutTokens: out.Usage.OutputTokens}, nil
}

// ---------- OpenAI (cloud, risposta completa) ----------

type openaiClient struct {
	key, model string
}

func (c *openaiClient) Chat(ctx context.Context, msgs []Message, onToken func(string)) (Reply, error) {
	body, _ := json.Marshal(map[string]any{
		"model":    c.model,
		"messages": msgs,
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, "https://api.openai.com/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return Reply{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.key)

	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := doJSON(req, &out, []string{c.key}); err != nil {
		return Reply{}, err
	}
	if len(out.Choices) == 0 {
		return Reply{}, errors.New("risposta OpenAI senza contenuto")
	}
	text := out.Choices[0].Message.Content
	onToken(text)
	return Reply{Text: text, InTokens: out.Usage.PromptTokens, OutTokens: out.Usage.CompletionTokens}, nil
}

// doJSON esegue la richiesta, controlla lo stato e decodifica il JSON. Le chiavi non compaiono mai negli errori.
func doJSON(req *http.Request, out any, secrets []string) error {
	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("richiesta fallita: %s", redact(err.Error(), secrets))
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("il provider ha risposto %d: %s", resp.StatusCode, redact(clip(string(raw), 500), secrets))
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return fmt.Errorf("risposta non valida: %w", err)
	}
	return nil
}

// ============================================================================
// Conversazione
// ============================================================================

// Session mantiene la conversazione con un limite di messaggi. Il system prompt non viene mai rimosso.
type Session struct {
	system Message
	msgs   []Message
	max    int
}

func NewSession(system string, max int) *Session {
	return &Session{system: Message{Role: "system", Content: system}, max: max}
}

// Add aggiunge un messaggio. Due messaggi consecutivi dello stesso ruolo vengono uniti,
// così la sequenza resta valida per tutti i provider.
func (s *Session) Add(role, content string) {
	if content == "" {
		return
	}
	if n := len(s.msgs); n > 0 && s.msgs[n-1].Role == role {
		s.msgs[n-1].Content += "\n\n" + content
	} else {
		s.msgs = append(s.msgs, Message{Role: role, Content: content})
	}
	s.trim()
}

// Pop rimuove l'ultimo messaggio, usato quando una chiamata al modello fallisce.
func (s *Session) Pop() {
	if n := len(s.msgs); n > 0 {
		s.msgs = s.msgs[:n-1]
	}
}

func (s *Session) Messages() []Message {
	out := make([]Message, 0, len(s.msgs)+1)
	out = append(out, s.system)
	return append(out, s.msgs...)
}

func (s *Session) trim() {
	if s.max > 0 && len(s.msgs) > s.max {
		s.msgs = s.msgs[len(s.msgs)-s.max:]
	}
	// La conversazione deve ripartire da un messaggio dell'utente.
	for len(s.msgs) > 0 && s.msgs[0].Role != "user" {
		s.msgs = s.msgs[1:]
	}
}

// ============================================================================
// Riconoscimento di blocchi shell e file proposti
// ============================================================================

type FileBlock struct {
	Path    string
	Content string
}

type CodeBlock struct {
	Lang string
	Body string
}

func extractFileBlocks(text string) []FileBlock {
	var out []FileBlock
	for _, m := range fileBlockRe.FindAllStringSubmatch(text, -1) {
		out = append(out, FileBlock{Path: m[1], Content: m[2]})
	}
	return out
}

// extractShellBlocks restituisce solo i blocchi dichiarati come shell.
// I blocchi FILE vengono esclusi prima, così un file .sh non viene mai eseguito.
func extractShellBlocks(text string) []CodeBlock {
	stripped := fileBlockRe.ReplaceAllString(text, "")
	var out []CodeBlock
	for _, m := range codeBlockRe.FindAllStringSubmatch(stripped, -1) {
		lang := strings.ToLower(m[1])
		body := strings.TrimSpace(m[2])
		if shellLangs[lang] && body != "" {
			out = append(out, CodeBlock{Lang: lang, Body: body})
		}
	}
	return out
}

// ============================================================================
// Percorsi sicuri e salvataggio dei file
// ============================================================================

type plannedFile struct {
	Rel     string
	Abs     string
	Content string
	Exists  bool
	Err     error
}

// cleanRelPath accetta solo percorsi relativi che restano dentro la radice.
func cleanRelPath(rel string) (string, error) {
	if rel == "" {
		return "", errors.New("percorso vuoto")
	}
	if filepath.IsAbs(rel) || strings.HasPrefix(rel, "/") || strings.HasPrefix(rel, "~") {
		return "", errors.New("percorso assoluto non consentito")
	}
	clean := filepath.Clean(filepath.FromSlash(rel))
	if clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", errors.New("percorso fuori dalla cartella di progetto")
	}
	return clean, nil
}

// resolveInside controlla il percorso componente per componente e rifiuta i symlink.
func resolveInside(root, rel string) (string, error) {
	clean, err := cleanRelPath(rel)
	if err != nil {
		return "", err
	}
	cur := root
	for _, part := range strings.Split(clean, string(filepath.Separator)) {
		cur = filepath.Join(cur, part)
		info, err := os.Lstat(cur)
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) {
				continue // il resto del percorso non esiste ancora
			}
			return "", err
		}
		if info.Mode()&fs.ModeSymlink != 0 {
			return "", fmt.Errorf("symlink non consentito: %s", part)
		}
	}
	abs := filepath.Join(root, clean)
	r, err := filepath.Rel(root, abs)
	if err != nil || r == ".." || strings.HasPrefix(r, ".."+string(filepath.Separator)) {
		return "", errors.New("percorso fuori dalla cartella di progetto")
	}
	return abs, nil
}

func withTimestamp(path string) string {
	ext := filepath.Ext(path)
	return strings.TrimSuffix(path, ext) + "-" + time.Now().Format("20060102-150405") + ext
}

func writeFile(path, content string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(content), 0o644)
}

func (a *App) askConflict(rel string) string {
	ans := strings.ToLower(a.ask(fmt.Sprintf("  %s esiste già. [o] sovrascrivi, [c] copia con data, [s] salta (predefinito s): ", rel)))
	switch ans {
	case "o":
		return "o"
	case "c":
		return "c"
	}
	return "s"
}

// handleFiles mostra l'anteprima dei file proposti e scrive solo dopo conferma.
func (a *App) handleFiles(blocks []FileBlock) {
	if len(blocks) == 0 {
		return
	}
	plans := make([]plannedFile, 0, len(blocks))
	for _, b := range blocks {
		p := plannedFile{Rel: b.Path, Content: b.Content}
		abs, err := resolveInside(a.projectDir, b.Path)
		if err == nil && len(b.Content) > maxFileBytes {
			err = fmt.Errorf("file troppo grande (oltre %d byte)", maxFileBytes)
		}
		p.Abs, p.Err = abs, err
		if err == nil {
			if _, statErr := os.Lstat(abs); statErr == nil {
				p.Exists = true
			}
		}
		plans = append(plans, p)
	}

	fmt.Fprintf(a.out, "\n%s[?] Il modello propone %d file in %s%s\n", yellow, len(plans), a.projectDir, reset)
	valid := 0
	for _, p := range plans {
		if p.Err != nil {
			fmt.Fprintf(a.out, "  %s[RIFIUTATO]%s %s: %v\n", red, reset, p.Rel, p.Err)
			a.logf("file_rifiutato", p.Rel+": "+p.Err.Error())
			continue
		}
		state := "nuovo"
		if p.Exists {
			state = "esiste"
		}
		fmt.Fprintf(a.out, "  [%s] %s (%d byte)\n", state, p.Rel, len(p.Content))
		valid++
	}
	if valid == 0 {
		return
	}
	if !a.confirm(fmt.Sprintf("Salvare i %d file validi?", valid)) {
		a.logf("file_annullati", fmt.Sprintf("%d file", valid))
		fmt.Fprintf(a.out, "%s[*] Nessun file salvato.%s\n", gray, reset)
		return
	}

	for _, p := range plans {
		if p.Err != nil {
			continue
		}
		target := p.Abs
		if p.Exists {
			switch a.askConflict(p.Rel) {
			case "o":
			case "c":
				target = withTimestamp(p.Abs)
			default:
				a.logf("file_saltato", p.Rel)
				fmt.Fprintf(a.out, "  %s[*] saltato%s\n", gray, reset)
				continue
			}
		}
		if err := writeFile(target, p.Content); err != nil {
			fmt.Fprintf(a.out, "  %s[!] errore su %s: %v%s\n", red, p.Rel, err, reset)
			a.logf("errore_file", err.Error())
			continue
		}
		a.logf("file_creato", target)
		fmt.Fprintf(a.out, "  %s[+] salvato: %s%s\n", green, target, reset)
	}
}

// ============================================================================
// Esecuzione dei comandi
// ============================================================================

func shellBinary(lang string) string {
	if lang == "sh" {
		return "/bin/sh"
	}
	return "/bin/bash"
}

// filteredEnv rimuove le chiavi API dall'ambiente dei comandi, così il modello non può leggerle.
func filteredEnv() []string {
	drop := map[string]bool{"ANTHROPIC_API_KEY": true, "OPENAI_API_KEY": true}
	var env []string
	for _, kv := range os.Environ() {
		name, _, _ := strings.Cut(kv, "=")
		if !drop[name] {
			env = append(env, kv)
		}
	}
	return env
}

// runShell esegue il comando nella cartella di progetto. Restituisce output (stdout+stderr) e codice di uscita.
func runShell(ctx context.Context, timeout time.Duration, lang, command, dir string) (string, int, error) {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	bin := shellBinary(lang)
	if _, err := os.Stat(bin); err != nil {
		bin = "/bin/sh"
	}
	cmd := exec.CommandContext(ctx, bin, "-c", command)
	cmd.Dir = dir
	cmd.Env = filteredEnv()

	out, err := cmd.CombinedOutput()
	if err == nil {
		return string(out), 0, nil
	}
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		return string(out), ee.ExitCode(), nil
	}
	return string(out), -1, err
}

// runApprovedCommand mantiene una sola conferma lato utente e poi sceglie
// esplicitamente il backend: HexStrike se -s è presente, shell locale altrimenti.
func (a *App) runApprovedCommand(ctx context.Context, lang, command string) (string, int, error) {
	if a.hexstrike != nil {
		return a.hexstrike.command(ctx, command)
	}
	return runShell(ctx, a.cfg.ShellTimeout(), lang, command, a.projectDir)
}

// ============================================================================
// Registro delle azioni (JSON Lines, file privato)
// ============================================================================

type Logger struct {
	path string
}

type logEntry struct {
	Time   string `json:"time"`
	Event  string `json:"event"`
	Detail string `json:"detail"`
}

func (l *Logger) Write(event, detail string) {
	if l == nil || l.path == "" {
		return
	}
	if r := []rune(detail); len(r) > 4000 {
		detail = string(r[:4000]) + " [troncato]"
	}
	if err := os.MkdirAll(filepath.Dir(l.path), 0o700); err != nil {
		fmt.Fprintf(os.Stderr, "log non disponibile: %v\n", err)
		return
	}
	f, err := os.OpenFile(l.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		fmt.Fprintf(os.Stderr, "log non disponibile: %v\n", err)
		return
	}
	defer f.Close()
	b, _ := json.Marshal(logEntry{Time: time.Now().Format(time.RFC3339), Event: event, Detail: detail})
	f.Write(append(b, '\n'))
}

// ============================================================================
// Flusso dell'applicazione
// ============================================================================

type App struct {
	cfg        *Config
	in         *bufio.Reader
	out        io.Writer
	client     Client
	session    *Session
	log        *Logger
	projectDir string
	secrets    []string
	spent      int
	hexstrike  *hexStrikeClient
}

// ask legge una riga dal lettore condiviso: è l'unico punto di lettura da tastiera.
func (a *App) ask(prompt string) string {
	fmt.Fprint(a.out, prompt)
	line, _ := a.in.ReadString('\n')
	return strings.TrimSpace(line)
}

// confirm: il default è sempre "no". Con input da pipe la risposta è vuota, quindi nulla viene eseguito.
func (a *App) confirm(prompt string) bool {
	ans := strings.ToLower(a.ask(prompt + " (s/N): "))
	return ans == "s" || ans == "si" || ans == "sì" || ans == "y" || ans == "yes"
}

func (a *App) logf(event, detail string) {
	a.log.Write(event, redact(detail, a.secrets))
}

func (a *App) confirmCloud() bool {
	fmt.Fprintf(a.out, "%s[!] Modalità cloud attiva: provider %s, modello %s.%s\n", yellow, a.cfg.Provider, a.cfg.Model, reset)
	fmt.Fprintln(a.out, "    Ogni messaggio della conversazione, incluso l'output dei comandi che approvi,")
	fmt.Fprintln(a.out, "    viene inviato al fornitore e lascia la tua macchina.")
	return a.confirm("Continuare?")
}

// turn gestisce uno scambio completo: domanda, risposta, eventuali file e comandi.
func (a *App) turn(ctx context.Context, input string) {
	a.logf("domanda", input)

	if a.cfg.Cloud && a.cfg.SpendCapTokens > 0 && a.spent >= a.cfg.SpendCapTokens {
		fmt.Fprintf(a.out, "%s[!] Tetto di token raggiunto (%d). Aumentalo in config.json o usa Ollama.%s\n", red, a.cfg.SpendCapTokens, reset)
		return
	}

	a.session.Add("user", input)
	tctx, cancel := context.WithTimeout(ctx, a.cfg.Timeout())
	defer cancel()

	fmt.Fprintln(a.out)
	reply, err := a.client.Chat(tctx, a.session.Messages(), func(tok string) { fmt.Fprint(a.out, tok) })
	fmt.Fprintln(a.out)
	if err != nil {
		a.session.Pop()
		fmt.Fprintf(a.out, "%s[!] %s%s\n", red, redact(err.Error(), a.secrets), reset)
		a.logf("errore", err.Error())
		return
	}

	a.spent += reply.InTokens + reply.OutTokens
	a.logf("risposta", reply.Text)
	a.logf("token", fmt.Sprintf("in=%d out=%d totale_sessione=%d", reply.InTokens, reply.OutTokens, a.spent))
	a.session.Add("assistant", reply.Text)

	a.handleFiles(extractFileBlocks(reply.Text))
	a.handleShell(ctx, reply.Text)
}

// handleShell mostra ogni blocco shell con tutte le informazioni e chiede conferma per ciascuno.
func (a *App) handleShell(ctx context.Context, text string) {
	blocks := extractShellBlocks(text)
	for i, b := range blocks {
		fmt.Fprintf(a.out, "\n%s[?] Blocco shell %d di %d%s\n", yellow, i+1, len(blocks), reset)
		fmt.Fprintf(a.out, "    Shell:    %s\n", shellBinary(b.Lang))
		if a.hexstrike != nil {
			fmt.Fprintf(a.out, "    Backend:  HexStrike %s\n", a.hexstrike.baseURL)
		} else {
			fmt.Fprintf(a.out, "    Cartella: %s\n", a.projectDir)
		}
		fmt.Fprintf(a.out, "    Comando:\n%s\n", indent(b.Body))

		if !a.confirm("Eseguire questo blocco?") {
			a.logf("comando_rifiutato", b.Body)
			fmt.Fprintf(a.out, "%s[*] Non eseguito.%s\n", gray, reset)
			continue
		}
		a.logf("comando_approvato", b.Body)
		if a.cfg.DryRun {
			fmt.Fprintf(a.out, "%s[dry-run] Comando non eseguito.%s\n", gray, reset)
			continue
		}

		fmt.Fprintf(a.out, "%s[>] Esecuzione in corso...%s\n", green, reset)
		out, code, err := a.runApprovedCommand(ctx, b.Lang, b.Body)
		if err != nil {
			fmt.Fprintf(a.out, "%s[!] Comando non avviato: %v%s\n", red, err, reset)
			a.logf("errore_comando", err.Error())
			a.session.Add("user", "[Comando non avviato: "+err.Error()+"]")
			continue
		}
		out = clip(out, maxOutputRunes)
		fmt.Fprintf(a.out, "%s--- output (codice di uscita %d) ---%s\n%s\n", gray, code, reset, strings.TrimRight(out, "\n"))
		a.logf("risultato", fmt.Sprintf("codice=%d\n%s", code, out))
		// L'output approvato torna alla conversazione, così puoi chiedere spiegazioni.
		a.session.Add("user", fmt.Sprintf("[Output del comando approvato]\n%s\n[Codice di uscita: %d]", out, code))
	}
}

func (a *App) repl(ctx context.Context) {
	fmt.Fprint(a.out, banner())
	fmt.Fprintf(a.out, "%s  modello %s%s  %s·  provider %s%s\n", gray, a.cfg.Model, reset, gray, a.cfg.Provider, reset)
	fmt.Fprintf(a.out, "%s  cartella %s%s\n", gray, a.projectDir, reset)
	fmt.Fprintf(a.out, "%s  scrivi %sexit%s%s per uscire%s\n\n", gray, green, reset, gray, reset)
	for {
		fmt.Fprintf(a.out, "%sjx82 ❯%s ", green, reset)
		line, err := a.in.ReadString('\n')
		input := strings.TrimSpace(line)
		if input == "exit" || input == "quit" {
			break
		}
		if input != "" {
			a.turn(ctx, input)
		}
		if err != nil {
			break
		}
	}
	fmt.Fprintln(a.out, "[*] Sessione chiusa.")
}

func indent(s string) string {
	lines := strings.Split(s, "\n")
	for i := range lines {
		lines[i] = "      " + lines[i]
	}
	return strings.Join(lines, "\n")
}

func clip(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "\n[output troncato]"
}

// minDuration evita un timeout di health-check più lungo del timeout globale.
func minDuration(a, b time.Duration) time.Duration {
	if a < b {
		return a
	}
	return b
}

// ============================================================================
// Avvio, flag e lettura dell'input
// ============================================================================

func main() {
	initColors()
	cfg := defaultConfig()
	if err := loadConfigFile(cfg); err != nil {
		fatal(err)
	}
	parseFlags(cfg)
	if err := cfg.resolve(); err != nil {
		fatal(err)
	}

	var hexstrike *hexStrikeClient
	if cfg.HexStrikeURL != "" {
		var err error
		hexstrike, err = newHexStrikeClient(cfg.HexStrikeURL, cfg.Timeout())
		if err != nil {
			fatal(err)
		}
		hctx, cancel := context.WithTimeout(context.Background(), minDuration(cfg.Timeout(), 10*time.Second))
		health, err := hexstrike.health(hctx)
		cancel()
		if err != nil {
			fatal(err)
		}
		fmt.Fprintf(os.Stderr, "%s[+] HexStrike connesso: %s (status=%v)\n%s", green, hexstrike.baseURL, health["status"], reset)
	}

	client, secrets, err := newClient(cfg)
	if err != nil {
		fatal(err)
	}

	project := cfg.Project
	if project == "" {
		project = "sessione-" + time.Now().Format("20060102-150405")
	}
	if strings.ContainsAny(project, `/\`) || project == "." || project == ".." {
		fatal(errors.New("nome progetto non valido"))
	}
	projectDir := filepath.Join(cfg.WorkRoot, project)
	if err := os.MkdirAll(projectDir, 0o755); err != nil {
		fatal(err)
	}

	in := bufio.NewReader(os.Stdin)
	app := &App{
		cfg:        cfg,
		in:         in,
		out:        os.Stdout,
		client:     client,
		session:    NewSession(loadPrompt(cfg), cfg.MaxMessages),
		log:        &Logger{path: cfg.LogPath},
		projectDir: projectDir,
		secrets:    secrets,
		hexstrike:  hexstrike,
	}
	app.logf("avvio", fmt.Sprintf("provider=%s modello=%s progetto=%s", cfg.Provider, cfg.Model, project))

	ctx := context.Background()

	if cfg.Interactive {
		if cfg.Cloud && !app.confirmCloud() {
			fmt.Println("Annullato.")
			return
		}
		app.repl(ctx)
		return
	}

	input, piped, err := oneShotInput(in, strings.Join(flag.Args(), " "))
	if err != nil {
		fatal(err)
	}
	if piped && cfg.Cloud {
		fatal(errors.New("con -a non è supportato l'input da pipe: lancia il programma da terminale"))
	}
	if cfg.Cloud && !app.confirmCloud() {
		fmt.Println("Annullato.")
		return
	}
	app.turn(ctx, input)
}

// oneShotInput legge la domanda dagli argomenti e/o dalla pipe. Restituisce anche se l'input arriva da pipe.
func oneShotInput(in *bufio.Reader, question string) (string, bool, error) {
	stat, err := os.Stdin.Stat()
	piped := err == nil && (stat.Mode()&os.ModeCharDevice) == 0
	if !piped {
		if question == "" {
			return "", false, errUsage
		}
		return question, false, nil
	}
	data, err := io.ReadAll(io.LimitReader(in, maxPipeBytes))
	if err != nil {
		return "", true, err
	}
	content := strings.TrimSpace(string(data))
	switch {
	case content == "" && question == "":
		return "", true, errUsage
	case question == "":
		return content, true, nil
	case content == "":
		return question, true, nil
	default:
		return "Contesto:\n" + content + "\n\nIstruzione: " + question, true, nil
	}
}

func parseFlags(cfg *Config) {
	flag.StringVar(&cfg.Model, "m", cfg.Model, "modello da usare")
	flag.BoolVar(&cfg.Interactive, "i", false, "modalità interattiva")
	flag.BoolVar(&cfg.Cloud, "a", false, "usa un provider cloud (vedi -provider)")
	flag.StringVar(&cfg.Provider, "provider", cfg.Provider, "provider cloud: anthropic o openai (con -a)")
	flag.BoolVar(&cfg.DryRun, "n", false, "dry-run: mostra i comandi senza eseguirli")
	flag.StringVar(&cfg.Project, "p", "", "nome del progetto (cartella di lavoro)")
	flag.StringVar(&cfg.WorkRoot, "w", cfg.WorkRoot, "cartella radice dei progetti")
	flag.StringVar(&cfg.HexStrikeURL, "s", "", "server HexStrike, es. 127.0.0.1:8888 o http://127.0.0.1:8080")
	flag.Usage = usage
	flag.Parse()
}

func banner() string {
	return fmt.Sprintf("\n%s%s  ◆ JX82 GPT%s  %sassistente da terminale%s\n%s  ────────────────────────────────%s\n",
		bold, cyan, reset, gray, reset, gray, reset)
}

func section(w io.Writer, title string) {
	fmt.Fprintf(w, "\n%s%s%s %s\n", bold, yellow, title, reset)
}

func usage() {
	w := os.Stderr
	fmt.Fprint(w, banner())

	section(w, "COME SI USA")
	fmt.Fprintf(w, "  %sjx82gpt \"domanda\"%s              fai una domanda\n", green, reset)
	fmt.Fprintf(w, "  %sjx82gpt -i%s                       chat continua da terminale\n", green, reset)
	fmt.Fprintf(w, "  %scat file.log | jx82gpt \"...\"%s     analizza un file o un output\n", green, reset)

	section(w, "OPZIONI")
	fmt.Fprintf(w, "  %s-n%s           simulazione: mostra i comandi senza eseguirli\n", green, reset)
	fmt.Fprintf(w, "  %s-p%s           nome del progetto (dove vengono salvati i file)\n", green, reset)
	fmt.Fprintf(w, "  %s-m%s           modello da usare\n", green, reset)
	fmt.Fprintf(w, "  %s-a%s           usa un modello cloud invece di quello locale\n", green, reset)
	fmt.Fprintf(w, "  %s-provider%s    anthropic o openai (insieme a -a)\n", green, reset)
	fmt.Fprintf(w, "  %s-w%s           cartella generale dei progetti\n", green, reset)
	fmt.Fprintf(w, "  %s-s%s           server HexStrike (es. 127.0.0.1:8888)\n", green, reset)

	section(w, "SICUREZZA")
	fmt.Fprintf(w, "  %s✓%s nulla viene eseguito senza il tuo sì\n", green, reset)
	fmt.Fprintf(w, "  %s✓%s nessun file viene scritto senza anteprima\n", green, reset)
	fmt.Fprintf(w, "  %s✓%s le chiavi API non vengono mai mostrate\n", green, reset)

	section(w, "DOVE SONO I TUOI DATI")
	fmt.Fprintf(w, "  %sconfig%s  ~/.jx82mygpt/config.json\n", gray, reset)
	fmt.Fprintf(w, "  %sprompt%s  ~/.jx82mygpt/prompt.txt\n", gray, reset)
	fmt.Fprintf(w, "  %slog%s     ~/.jx82mygpt/log.jsonl\n", gray, reset)

	section(w, "RISPOSTE")
	fmt.Fprintf(w, "  %ss%s = sì     %sinvio%s = no\n\n", green, reset, gray, reset)
}

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "%s[!] %v%s\n", red, err, reset)
	os.Exit(1)
}

