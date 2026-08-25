package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
)

// bridgeEnv talks JSONL to bridge/agent_bridge.py over stdin/stdout.
type bridgeEnv struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Scanner
	mu     sync.Mutex
	game   string
}

func openBridgeEnv(game, bridgePy string) (Env, error) {
	if bridgePy == "" {
		bridgePy = filepath.Join("bridge", "agent_bridge.py")
	}
	if _, err := os.Stat(bridgePy); err != nil {
		return nil, fmt.Errorf("bridge script %s: %w", bridgePy, err)
	}
	cmd := exec.Command("python3", bridgePy, "--game", game)
	cmd.Env = os.Environ()
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		_ = stdin.Close()
		return nil, err
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		_ = stdin.Close()
		return nil, fmt.Errorf("start bridge: %w", err)
	}
	sc := bufio.NewScanner(stdout)
	sc.Buffer(make([]byte, 0, 256*1024), 4*1024*1024)
	return &bridgeEnv{cmd: cmd, stdin: stdin, stdout: sc, game: game}, nil
}

func (b *bridgeEnv) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	_ = b.write(map[string]any{"op": "close"})
	_ = b.stdin.Close()
	if b.cmd != nil && b.cmd.Process != nil {
		_ = b.cmd.Wait()
	}
	return nil
}

func (b *bridgeEnv) Reset() (Frame, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.write(map[string]any{"op": "reset"}); err != nil {
		return Frame{}, err
	}
	return b.readFrame()
}

func (b *bridgeEnv) Step(a Action) (Frame, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err := b.write(map[string]any{
		"op": "step",
		"action": map[string]any{
			"id": a.ID,
			"x":  a.X,
			"y":  a.Y,
		},
	}); err != nil {
		return Frame{}, err
	}
	return b.readFrame()
}

func (b *bridgeEnv) OracleAction(Frame) int { return -1 }

func (b *bridgeEnv) write(v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	_, err = b.stdin.Write(data)
	return err
}

func (b *bridgeEnv) readFrame() (Frame, error) {
	if !b.stdout.Scan() {
		if err := b.stdout.Err(); err != nil {
			return Frame{}, err
		}
		return Frame{}, fmt.Errorf("bridge: EOF")
	}
	var resp struct {
		OK    bool   `json:"ok"`
		Error string `json:"error"`
		Frame Frame  `json:"frame"`
	}
	if err := json.Unmarshal(b.stdout.Bytes(), &resp); err != nil {
		return Frame{}, err
	}
	if !resp.OK {
		if resp.Error == "" {
			resp.Error = "bridge error"
		}
		return Frame{}, fmt.Errorf("%s", resp.Error)
	}
	if resp.Frame.GameID == "" {
		resp.Frame.GameID = b.game
	}
	return resp.Frame, nil
}
