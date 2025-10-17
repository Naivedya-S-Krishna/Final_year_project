// suggestions.js
function clamp(v, a = 0, b = 1) {
  return Math.max(a, Math.min(b, v));
}

function getSeverity(score) {
  if (score >= 0.75) return "high";
  if (score >= 0.45) return "moderate";
  return "low";
}

function analyzeReport(report) {
  const r = report || {};

  const stress = clamp(Number(r.stress_score) || 0);
  const voiceEmotion = (r.voice && r.voice.emotion) || "neutral";
  const faceEmotion = (r.face && r.face.emotion) || "neutral";
  const textSentiment =
    (r.text && r.text.sentiment) ||
    (r.text && r.text.score < 0 ? "negative" : "positive") ||
    "neutral";

  const negativeEmotions = ["sad", "angry", "fear", "anxious"];
  let emotionScore = 0;
  if (negativeEmotions.includes(voiceEmotion)) emotionScore += 0.4;
  if (negativeEmotions.includes(faceEmotion)) emotionScore += 0.4;
  if (textSentiment === "negative") emotionScore += 0.3;

  const overallScore = clamp(stress * 0.7 + emotionScore * 0.3);
  const severity = getSeverity(overallScore);
  const suggestions = [];

  if (severity === "high") {
    suggestions.push("High stress detected. Try deep breathing or a short break.");
    suggestions.push("Consider reaching out to a friend or professional.");
  } else if (severity === "moderate") {
    suggestions.push("Moderate stress detected. Take short breaks or go for a walk.");
  } else {
    suggestions.push("Low stress detected. Keep maintaining your current routine.");
  }

  return {
    severity,
    overallScore,
    summary: `Stress: ${stress}, Voice: ${voiceEmotion}, Face: ${faceEmotion}, Sentiment: ${textSentiment}`,
    suggestions,
  };
}

module.exports = { analyzeReport };
