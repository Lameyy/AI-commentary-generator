# AI-commentary-generator
This is an AI based commentary generator created with the uses of  Gemini API, Azure Open API, HTML for front-end, and Flask for backend.
1. Initial web-based interface to accept user input for video.
2. Video gets processed using Flask as backend and sent to the Gemini API.
3. Gemini API understands the video, and gives a word of what's happening there.
4. Azure openAi API broadcasts the prompt of Gemini API as a professional commentator.
5. gTTS(Gemini Text-to-speech) then process the prompt to the commentary which is then displayed in the front-end again
