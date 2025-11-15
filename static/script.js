const analyzeBtn = document.getElementById("analyzeBtn");
const textInput = document.getElementById("textInput");
const resultDiv = document.getElementById("result");
const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const csvResult = document.getElementById("csvResult");
const summaryDiv = document.getElementById("summary");
const wordcloudImg = document.getElementById("wordcloud");

// ---- Single text analysis ----
analyzeBtn.addEventListener("click", async () => {
  const text = textInput.value.trim();
  if (!text) {
    alert("Please enter some text!");
    return;
  }

  resultDiv.innerHTML = "⏳ Analyzing...";
  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const data = await res.json();
  resultDiv.innerHTML = `<b>Sentiment:</b> ${data.predicted_sentiment}`;
});

// ---- CSV Upload ----
uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    alert("Please select a CSV file!");
    return;
  }

  csvResult.innerHTML = "⏳ Processing file...";
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch("/predict", { method: "POST", body: formData });
  const data = await res.json();

  csvResult.innerHTML = `
    <p><b>Sentiment counts:</b> ${JSON.stringify(data.counts)}</p>
    <p><b>Summary:</b> ${data.summary}</p>
    <p><a href="data:text/csv;base64,${data.csv_file}" download="results.csv">📥 Download Results CSV</a></p>
  `;

  if (data.wordcloud) {
    wordcloudImg.src = "data:image/png;base64," + data.wordcloud;
  }
});
