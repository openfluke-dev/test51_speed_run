package main

import (
	"fmt"
	"math/rand"
)

// Frame mirrors ARC-AGI-3 FrameData shape used by the agents repo.
type Frame struct {
	Layers             [][][]int `json:"layers"` // C×H×W color ints 0..9
	State              string    `json:"state"`
	LevelsCompleted    int       `json:"levels_completed"`
	WinLevels          int       `json:"win_levels"`
	AvailableActions   []int     `json:"available_actions"`
	GameID             string    `json:"game_id"`
}

// Action is RESET (0) or ACTION1..7; ACTION6 may carry (X,Y) in 0..63.
type Action struct {
	ID int `json:"id"`
	X  int `json:"x"`
	Y  int `json:"y"`
}

// Env is the ARC-AGI-3-shaped game interface.
type Env interface {
	Reset() (Frame, error)
	Step(Action) (Frame, error)
	Close() error
	// OracleAction returns a heuristic best discrete action for Acc measuring (mock).
	// Bridge returns -1 when unknown.
	OracleAction(Frame) int
}

const (
	stateNotPlayed   = "NOT_PLAYED"
	stateNotFinished = "NOT_FINISHED"
	stateWin         = "WIN"
	stateGameOver    = "GAME_OVER"
)

// MockEnv is a Go-native chase-the-goal playfield (64×64, one layer).
// ACTION1=up ACTION2=down ACTION3=left ACTION4=right ACTION5=wait
// ACTION6=teleport click ACTION7=noop-alt RESET=0.
type MockEnv struct {
	rng       *rand.Rand
	gameID    string
	winLevels int
	levels    int
	ax, ay    int
	gx, gy    int
	steps     int
	maxSteps  int
}

func NewMockEnv(seed int64, gameID string) *MockEnv {
	if gameID == "" {
		gameID = "mock-ls"
	}
	return &MockEnv{
		rng:       rand.New(rand.NewSource(seed)),
		gameID:    gameID,
		winLevels: 3,
		maxSteps:  80,
	}
}

func (m *MockEnv) Close() error { return nil }

func (m *MockEnv) Reset() (Frame, error) {
	m.levels = 0
	m.steps = 0
	m.spawn()
	return m.frame(stateNotFinished), nil
}

func (m *MockEnv) spawn() {
	m.ax = 8 + m.rng.Intn(frameSize-16)
	m.ay = 8 + m.rng.Intn(frameSize-16)
	for {
		m.gx = 8 + m.rng.Intn(frameSize-16)
		m.gy = 8 + m.rng.Intn(frameSize-16)
		if abs(m.gx-m.ax)+abs(m.gy-m.ay) >= 12 {
			break
		}
	}
}

func (m *MockEnv) Step(a Action) (Frame, error) {
	if a.ID == 0 { // RESET
		m.spawn()
		m.steps = 0
		return m.frame(stateNotFinished), nil
	}
	m.steps++
	switch a.ID {
	case 1:
		m.ay = clamp(m.ay-1, 0, frameSize-1)
	case 2:
		m.ay = clamp(m.ay+1, 0, frameSize-1)
	case 3:
		m.ax = clamp(m.ax-1, 0, frameSize-1)
	case 4:
		m.ax = clamp(m.ax+1, 0, frameSize-1)
	case 5, 7:
		// wait / alt noop
	case 6:
		m.ax = clamp(a.X, 0, frameSize-1)
		m.ay = clamp(a.Y, 0, frameSize-1)
	}
	if m.ax == m.gx && m.ay == m.gy {
		m.levels++
		if m.levels >= m.winLevels {
			fr := m.frame(stateWin)
			return fr, nil
		}
		m.spawn()
		m.steps = 0
		return m.frame(stateNotFinished), nil
	}
	if m.steps >= m.maxSteps {
		m.spawn()
		m.steps = 0
		return m.frame(stateGameOver), nil
	}
	return m.frame(stateNotFinished), nil
}

func (m *MockEnv) OracleAction(fr Frame) int {
	dx := m.gx - m.ax
	dy := m.gy - m.ay
	if abs(dx)+abs(dy) == 0 {
		return 5
	}
	if abs(dx) >= abs(dy) {
		if dx < 0 {
			return 3
		}
		return 4
	}
	if dy < 0 {
		return 1
	}
	return 2
}

func (m *MockEnv) frame(state string) Frame {
	layer := make([][]int, frameSize)
	for y := 0; y < frameSize; y++ {
		row := make([]int, frameSize)
		layer[y] = row
	}
	layer[m.gy][m.gx] = 2 // goal
	layer[m.ay][m.ax] = 1 // agent
	return Frame{
		Layers:           [][][]int{layer},
		State:            state,
		LevelsCompleted:  m.levels,
		WinLevels:        m.winLevels,
		AvailableActions: []int{1, 2, 3, 4, 5, 6, 7},
		GameID:           m.gameID,
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func openEnv(game string, seed int64, bridgePy string) (Env, error) {
	game = trimSpace(game)
	if game == "" || game == "mock" {
		return NewMockEnv(seed, "mock-ls"), nil
	}
	return openBridgeEnv(game, bridgePy)
}

func trimSpace(s string) string {
	for len(s) > 0 && (s[0] == ' ' || s[0] == '\t') {
		s = s[1:]
	}
	for len(s) > 0 && (s[len(s)-1] == ' ' || s[len(s)-1] == '\t') {
		s = s[:len(s)-1]
	}
	return s
}

func envMustReset(e Env) Frame {
	fr, err := e.Reset()
	if err != nil {
		panic(fmt.Sprintf("reset: %v", err))
	}
	return fr
}
