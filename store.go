package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Store is a tide-inspired checkpoint: progress.json + history.json + results.json.
type Store struct {
	mu   sync.Mutex
	Root string
}

type Progress struct {
	Version   int         `json:"version"`
	UpdatedAt time.Time   `json:"updated_at"`
	Total     int         `json:"total"`
	NextIndex int         `json:"next_index"`
	DoneIDs   []string    `json:"done_ids"`
	BestID    string      `json:"best_id,omitempty"`
	BestScore float64     `json:"best_score,omitempty"`
	BestAcc   float64     `json:"best_acc,omitempty"`
	Current   string      `json:"current,omitempty"`
	Phase     string      `json:"phase,omitempty"`
	Completed []modeResult `json:"completed,omitempty"`
}

type HistoryPoint struct {
	At     time.Time `json:"at"`
	JobID  string    `json:"job_id"`
	Phase  string    `json:"phase"`
	Acc    float64   `json:"acc"`
	Score  float64   `json:"score"`
	Avail  float64   `json:"availability"`
	Levels int       `json:"levels"`
	LR     float64   `json:"lr"`
	Note   string    `json:"note,omitempty"`
}

func NewStore(root string) *Store {
	if root == "" {
		root = "test51_ckpt"
	}
	return &Store{Root: root}
}

func (s *Store) ensure() error {
	return os.MkdirAll(s.Root, 0o755)
}

func (s *Store) progressPath() string { return filepath.Join(s.Root, "progress.json") }
func (s *Store) historyPath() string  { return filepath.Join(s.Root, "history.json") }
func (s *Store) resultsPath() string  { return filepath.Join(s.Root, "results.json") }

func (s *Store) Load() (*Progress, []HistoryPoint, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensure(); err != nil {
		return nil, nil, err
	}
	var p Progress
	if b, err := os.ReadFile(s.progressPath()); err == nil {
		if err := json.Unmarshal(b, &p); err != nil {
			return nil, nil, err
		}
	} else if !os.IsNotExist(err) {
		return nil, nil, err
	}
	if p.Version == 0 {
		p.Version = 1
	}
	var hist []HistoryPoint
	if b, err := os.ReadFile(s.historyPath()); err == nil {
		_ = json.Unmarshal(b, &hist)
	}
	return &p, hist, nil
}

func (s *Store) SaveProgress(p *Progress) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensure(); err != nil {
		return err
	}
	p.UpdatedAt = time.Now().UTC()
	p.Version = 1
	b, err := json.MarshalIndent(p, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.progressPath() + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.progressPath())
}

func (s *Store) AppendHistory(h HistoryPoint) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensure(); err != nil {
		return err
	}
	var hist []HistoryPoint
	if b, err := os.ReadFile(s.historyPath()); err == nil {
		_ = json.Unmarshal(b, &hist)
	}
	hist = append(hist, h)
	b, err := json.MarshalIndent(hist, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.historyPath() + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.historyPath())
}

func (s *Store) SaveResults(v any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.ensure(); err != nil {
		return err
	}
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	tmp := s.resultsPath() + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, s.resultsPath())
}

// resultsFile is the on-disk shape of results.json (partial decode is fine).
type resultsFile struct {
	Results []modeResult `json:"results"`
	Reports []treeReport `json:"reports"`
	BestID  string       `json:"best_id"`
}

// LoadResults reads results.json if present (reports + full result rows).
func (s *Store) LoadResults() (resultsFile, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out resultsFile
	b, err := os.ReadFile(s.resultsPath())
	if err != nil {
		if os.IsNotExist(err) {
			return out, nil
		}
		return out, err
	}
	if err := json.Unmarshal(b, &out); err != nil {
		return out, err
	}
	return out, nil
}

func doneSet(ids []string) map[string]bool {
	m := make(map[string]bool, len(ids))
	for _, id := range ids {
		m[id] = true
	}
	return m
}

func mustWrite(path string, v any) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
	}
}
