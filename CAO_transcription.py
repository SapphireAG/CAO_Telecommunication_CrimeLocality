import whisper
model=whisper.load_model("base")
result = model.transcribe("/Users/amangolani/Downloads/CAO/BankScamAudio.mp3")
print(result["text"])

