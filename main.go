package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/invopop/jsonschema"
)

const (
	ANSI_BRIGHT_BLUE   = "\u001b[94m"
	ANSI_BRIGHT_YELLOW = "\u001b[93m"
	ANSI_RESET         = "\u001b[0m"
)

func main() {
	client := anthropic.NewClient()
	userMessageFn := UserMessage()
	tools := []ToolDefinition{ReadFileDefinition}
	agent := NewAgent(&client, userMessageFn, tools)
	if err := agent.Run(context.TODO()); err != nil {
		fmt.Printf("Error: %v\n", err)
	}
}

// UserMessage captures user input from the CLI and returns it via a closure
func UserMessage() func() (string, bool) {
	scanner := bufio.NewScanner(os.Stdin)

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
	tools          []ToolDefinition
}

// NewAgent creates a new instance of an Agent
func NewAgent(
	client *anthropic.Client,
	getUserMessage func() (string, bool),
	tools []ToolDefinition,
) *Agent {
	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
		tools:          tools,
	}
}

// Run starts a conversation with Claude
func (a *Agent) Run(ctx context.Context) error {
	conversation := []anthropic.MessageParam{}

	fmt.Println("Chat with Claude (use 'ctrl+C' to exit)")

	// Run a continuous capture sesssion for chatting with Claude
	readUserInput := true
	for {
		// Capture user input from the CLI
		if readUserInput {
			a.requestPrompt()
			userInput, ok := a.getUserMessage()
			if !ok {
				break
			}

			// convert user input to a message and append to conversation for contextual history or short term memory
			userMessage := anthropic.NewUserMessage(anthropic.NewTextBlock(userInput))
			conversation = append(conversation, userMessage)
		}

		// Run inference with the updated conversation, ala send the conversation to Claude
		message, err := a.runInference(ctx, conversation)
		if err != nil {
			return err
		}

		// Append Claude's response to the conversation history
		conversation = append(conversation, message.ToParam())

		// Print out Claude's response to the CLI
		toolResults := []anthropic.ContentBlockParamUnion{}
		for _, content := range message.Content {
			switch content.Type {
			case "text":
				a.responsePrompt(content.Text)
			case "tool_use":
				result := a.executeTool(content.ID, content.Name, content.Input)
				toolResults = append(toolResults, result)
			default:
				// Ignore non-text content for simplicity
			}
		}

		if len(toolResults) == 0 {
			readUserInput = true
			continue
		}
		readUserInput = false
		conversation = append(conversation, anthropic.NewUserMessage(toolResults...))
	}

	return nil
}

// Request prompt for user input
func (a *Agent) requestPrompt() {
	fmt.Printf("%sYou%s: ", ANSI_BRIGHT_BLUE, ANSI_RESET)
}

// Response prompt for Claude's output
func (a *Agent) responsePrompt(response string) {
	fmt.Printf("%sClaude%s: %s\n", ANSI_BRIGHT_YELLOW, ANSI_RESET, response)
}

// runInference sends the conversation history to Claude and returns the response
func (a *Agent) runInference(ctx context.Context, conversation []anthropic.MessageParam) (*anthropic.Message, error) {
	anthropicTools := []anthropic.ToolUnionParam{}
	for _, tool := range a.tools {
		anthropicTools = append(anthropicTools, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        tool.Name,
				Description: anthropic.String(tool.Description),
				InputSchema: tool.InputSchema,
			},
		})
	}

	return a.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.ModelClaude3_7SonnetLatest,
		MaxTokens: int64(1024),
		Messages:  conversation,
		Tools:     anthropicTools,
	})
}

func (a *Agent) executeTool(id, name string, input json.RawMessage) anthropic.ContentBlockParamUnion {
	var toolDef ToolDefinition
	var found bool
	for _, tool := range a.tools {
		if tool.Name == name {
			toolDef = tool
			found = true
			break
		}
	}
	if !found {
		return anthropic.NewToolResultBlock(id, "tool not found", true)
	}

	fmt.Printf("\u001b[92mtool\u001b[0m: %s(%s)\n", name, input)
	response, err := toolDef.Function(input)
	if err != nil {
		return anthropic.NewToolResultBlock(id, err.Error(), true)
	}
	return anthropic.NewToolResultBlock(id, response, false)
}

type ToolDefinition struct {
	Name        string                         `json:"name"`
	Description string                         `json:"description"`
	InputSchema anthropic.ToolInputSchemaParam `json:"input_schema"`
	Function    func(input json.RawMessage) (string, error)
}

var ReadFileDefinition = ToolDefinition{
	Name:        "read_file",
	Description: "Read the contents of a given relative file path. Use this when you want to see what's inside a file. Do not use this with directory names.",
	InputSchema: ReadFileInputSchema,
	Function:    ReadFile,
}

type ReadFileInput struct {
	Path string `json:"path" jsonschema_description:"The relative path of a file in the working directory."`
}

var ReadFileInputSchema = GenerateSchema[ReadFileInput]()

func ReadFile(input json.RawMessage) (string, error) {
	readFileInput := ReadFileInput{}
	err := json.Unmarshal(input, &readFileInput)
	if err != nil {
		panic(err)
	}

	content, err := os.ReadFile(readFileInput.Path)
	if err != nil {
		return "", err
	}
	return string(content), nil
}

func GenerateSchema[T any]() anthropic.ToolInputSchemaParam {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	var v T

	schema := reflector.Reflect(v)

	return anthropic.ToolInputSchemaParam{
		Properties: schema.Properties,
	}
}
