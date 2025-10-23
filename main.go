package main

import (
	"bufio"
	"context"
	"fmt"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
)

const (
	ANSI_BRIGHT_BLUE   = "\u001b[94m"
	ANSI_BRIGHT_YELLOW = "\u001b[93m"
	ANSI_RESET         = "\u001b[0m"
)

func main() {
	client := anthropic.NewClient()
	agent := NewAgent(&client, promptCaptureFunction())
	if err := agent.Run(context.TODO()); err != nil {
		fmt.Printf("Error: %v\n", err)
	}
}

func promptCaptureFunction() func() (string, bool) {
	scanner := bufio.NewScanner(os.Stdin)

	// Create an anonymous closure to capture the user prompt from the CLI
	return func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}

		return scanner.Text(), true
	}
}

type Agent struct {
	client         *anthropic.Client
	getUserMessage func() (string, bool)
}

func NewAgent(client *anthropic.Client, getUserMessage func() (string, bool)) *Agent {
	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
	}
}

func (a *Agent) Run(ctx context.Context) error {
	conversation := []anthropic.MessageParam{}

	fmt.Println("Chat with Claude (use 'ctrl+C' to exit)")

	// Run a continuous capture sesssion for a conversation with Claude
	for {
		// Capture user input from the CLI
		a.requestPrompt()
		userInput, ok := a.getUserMessage()
		if !ok {
			break
		}

		// convert user input to a message and append to conversation for contextual history or short term memory
		userMessage := anthropic.NewUserMessage(anthropic.NewTextBlock(userInput))
		conversation = append(conversation, userMessage)

		// Run inference with the updated conversation, ala send the conversation to Claude
		message, err := a.runInference(ctx, conversation)
		if err != nil {
			return err
		}

		// Append Claude's response to the conversation history
		conversation = append(conversation, message.ToParam())

		// Print out Claude's response to the CLI
		for _, content := range message.Content {
			switch content.Type {
			case "text":
				a.responsePrompt(content.Text)
			default:
				// Ignore non-text content for simplicity
			}
		}
	}

	return nil
}

func (a *Agent) requestPrompt() {
	fmt.Printf("%sYou%s: ", ANSI_BRIGHT_BLUE, ANSI_RESET)
}

func (a *Agent) responsePrompt(response string) {
	fmt.Printf("%sClaude%s: %s\n", ANSI_BRIGHT_YELLOW, ANSI_RESET, response)
}

func (a *Agent) runInference(ctx context.Context, conversation []anthropic.MessageParam) (*anthropic.Message, error) {
	return a.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.ModelClaude3_7SonnetLatest,
		MaxTokens: int64(1024),
		Messages:  conversation,
	})
}
