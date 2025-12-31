import Lean
import LeanCopilot.Models.Interface

set_option autoImplicit false

open Lean

namespace LeanCopilot


structure ExternalModel where
  name : String
  host : String := "localhost"
  port : UInt16 := 23337
deriving Inhabited, Repr


structure ExternalGenerator extends ExternalModel
deriving Repr


structure GeneratorRequest where
  name : String
  input : String
  «prefix» : String
deriving ToJson


structure Generation where
  output: String
  score: Float
deriving FromJson


structure GeneratorResponse where
  outputs : Array Generation
deriving FromJson


structure EnencoderRequest where
  name : String
  input : String
deriving ToJson


structure EncoderResponse where
  outputs : Array Float
deriving FromJson


def send {α β : Type} [ToJson α] [FromJson β] (req : α) (url : String) : IO β := do
  let reqStr := (toJson req).pretty 99999999999999999
  let out ← IO.Process.output {
    cmd := "curl"
    args := #["-X", "POST", url, "-H", "accept: application/json", "-H", "Content-Type: application/json", "-d", reqStr]
  }
  if out.exitCode != 0 then
     throw $ IO.userError s!"Request failed. Please check if the server is up at `{url}`."
  let some json := Json.parse out.stdout |>.toOption
    | throw $ IO.userError "Failed to parse response"
  let some res := (fromJson? json : Except String β) |>.toOption
    | throw $ IO.userError "Failed to parse response"
  return res


def ExternalGenerator.generate (model : ExternalGenerator) (input : String) (targetPrefix : String) : IO $ Array (String × Float) := do
  let url := s!"http://{model.host}:{model.port}/generate"
  let req : GeneratorRequest := {
    name := model.name,
    input := input,
    «prefix» := targetPrefix
  }
  let res : GeneratorResponse ← send req url
  return res.outputs.map fun g => (g.output, g.score)


def generateRunpod (input : String) (modelName : String) (targetPrefix : String) : IO $ Array (String × Float) := do
  let url := "https://router.huggingface.co/v1/chat/completions"
  
  -- Format the prompt (same as Python pre_process_input)
  let prompt := if modelName == "deepseek" then
    s!"Here is a theorem you need to prove in Lean:```lean\n{input}```\nNow you should suggest exactly one line tactic in lean code. Only output the tactic, no explanation, no comments, no theorem, nothing else:"
  else
    input
  
  -- Build JSON payload
  let message := Json.mkObj [("role", Json.str "user"), ("content", Json.str prompt)]
  let messages := Json.arr #[message]
  let reqJson := Json.mkObj [
    ("messages", messages),
    ("model", Json.str "deepseek-ai/DeepSeek-Prover-V2-671B:novita"),
    ("temperature", Json.num 0.6),
    ("max_tokens", Json.num 256),
    ("top_p", Json.num 0.9),
    ("n", Json.num 4)
  ]
  let reqStr := reqJson.pretty 99999999999999999
  
  -- Read HF_TOKEN from .env file in current directory
  let envFile := ".env"
  let envContent ← IO.FS.readFile envFile
  let lines := envContent.splitOn "\n"
  let mut token := ""
  for line in lines do
    if line.startsWith "HF_TOKEN=" then
      token := line.drop 9  -- Remove "HF_TOKEN=" prefix
  if token == "" then
    throw $ IO.userError "HF_TOKEN not found in .env file. Please create a .env file with: HF_TOKEN=your_token_here"
  
  -- Make API call
  let out ← IO.Process.output {
    cmd := "curl"
    args := #["-X", "POST", url, "-H", "accept: application/json", "-H", "Content-Type: application/json", "-H", s!"Authorization: Bearer {token}", "-d", reqStr]
  }
  if out.exitCode != 0 then
     throw $ IO.userError s!"Request failed. Please check if the API is accessible at `{url}`."
  
  -- Parse response
  let some json := Json.parse out.stdout |>.toOption
    | throw $ IO.userError "Failed to parse response"
  
  -- Debug: print raw response
  IO.println s!"Raw API response: {out.stdout}"
  
  let some choices := json.getObjVal? "choices" |>.toOption
    | throw $ IO.userError "No choices in response"
  let some choicesArray := choices.getArr? |>.toOption
    | throw $ IO.userError "Choices is not an array"
  
  -- Extract and clean responses (same as Python post_process_output)
  let mut results : Array (String × Float) := #[]
  for choice in choicesArray do
    let some message := choice.getObjVal? "message" |>.toOption
      | continue
    let some content := message.getObjVal? "content" |>.toOption
      | continue
    let some contentStr := content.getStr? |>.toOption
      | continue
    -- Clean the response (extract tactic from backticks)
    let cleaned := if modelName == "deepseek" then
      -- Extract content between backticks
      if contentStr.startsWith "`" && contentStr.endsWith "`" then
        contentStr.drop 1 |>.dropRight 1  -- Remove first and last backtick
      else
        contentStr
    else
      contentStr
    results := results.push (cleaned, 1.0)
  
  -- Deduplicate (same as Python choices_dedup)
  let mut uniqueData : List (String × Float) := []
  for result in results do
    if !uniqueData.any (·.1 == result.1) then
      uniqueData := uniqueData.append [result]
  
  return uniqueData.toArray


instance : TextToText ExternalGenerator := ⟨ExternalGenerator.generate⟩


structure ExternalEncoder extends ExternalModel
deriving Repr


def ExternalEncoder.encode (model : ExternalEncoder) (input : String) : IO FloatArray := do
  let url := s!"http://{model.host}:{model.port}/encode"
  let req : EnencoderRequest := {
    name := model.name,
    input := input,
  }
  let res : EncoderResponse ← send req url
  return FloatArray.mk res.outputs


instance : TextToVec ExternalEncoder := ⟨ExternalEncoder.encode⟩


end LeanCopilot
