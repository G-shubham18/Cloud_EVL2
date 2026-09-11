import re
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from config import MODALITY_ESTIMATOR_MODEL, DEVICE

class QuestionClassifier:
    def __init__(self):
        print(f"Loading Modality Estimator: {MODALITY_ESTIMATOR_MODEL}")
        self.model = SentenceTransformer(MODALITY_ESTIMATOR_MODEL, device=DEVICE)
        
        # Audio concept descriptions (speech, acoustic events, sounds, spoken words, music, ambient audio)
        self.audio_anchors = [
            "questions about spoken speech, dialog, words said, or spoken conversation",
            "questions about sound effects, background noise, audio events, music, or acoustic signals",
            "what noise or sound is heard in the video",
            "did someone say or speak something",
            "what did the speaker or person say",
            "who spoke or yelled in the audio track",
            "listening to music, sirens, laughter, applause, or environmental sounds",
            "audio transcript, voice, conversation, or speech utterance"
        ]
        
        # Visual concept descriptions (objects, subjects, colors, actions, spatial layout, physical appearance)
        self.visual_anchors = [
            "questions about visual appearance, colors, clothes, objects, or physical subjects",
            "questions about spatial layout, location, left, right, background, or visible scene",
            "what color is the shirt, vehicle, object, or item",
            "where is the person, object, or item positioned visually",
            "what action is visually being performed in the video frame",
            "who is visible or seen on screen in the video",
            "describe the visual scene, appearance, or background details",
            "visible text, physical objects on table, or optical keyframes",
            "how many people, persons, men, women, or individuals are in the video or visible in the scene",
            "how many chairs, tables, laptops, instruments, or objects appear or are counted in the video",
            "counting unique entities or visible count of objects and people"
        ]
        
        self.audio_embeds = self.model.encode(self.audio_anchors, convert_to_tensor=True)
        self.visual_embeds = self.model.encode(self.visual_anchors, convert_to_tensor=True)

        self.audio_keywords = {
            "say", "said", "spoke", "speaking", "talk", "talking", "sound", "noise",
            "listen", "heard", "hear", "music", "singing", "yell", "shout", "whisper",
            "voice", "applause", "laughter", "siren", "alarm", "speech", "dialogue"
        }

        self.visual_keywords = {
            "color", "wearing", "clothes", "shirt", "pants", "dress", "visible",
            "look", "see", "seen", "where", "behind", "next", "left", "right",
            "background", "foreground", "object", "car", "table", "holding", "standing",
            "how", "many", "count", "counting", "number", "total", "unique"
        }

    def estimate_beta(self, question: str) -> float:
        """
        Estimates the audio dependency weight beta(q) in [0, 1].
        1.0 means highly audio-dependent, 0.0 means highly visual-dependent.
        """
        q_lower = question.lower()
        words = set(re.findall(r'\b\w+\b', q_lower))

        # Direct keyword heuristic prior
        audio_kw_count = len(words.intersection(self.audio_keywords))
        visual_kw_count = len(words.intersection(self.visual_keywords))

        q_embed = self.model.encode([question], convert_to_tensor=True)
        
        # Mean + Max similarity to audio and visual semantic concepts
        audio_sims = cos_sim(q_embed, self.audio_embeds)[0]
        visual_sims = cos_sim(q_embed, self.visual_embeds)[0]

        audio_score = float((audio_sims.mean() * 0.5 + audio_sims.max() * 0.5).item())
        visual_score = float((visual_sims.mean() * 0.5 + visual_sims.max() * 0.5).item())

        # Adjust scores using keyword priors
        audio_score += audio_kw_count * 0.15
        visual_score += visual_kw_count * 0.15

        audio_score = max(0.001, audio_score)
        visual_score = max(0.001, visual_score)
        
        beta = audio_score / (audio_score + visual_score)
        
        # Clamp between 0 and 1
        beta = max(0.0, min(1.0, float(beta)))
        
        print(f"Question Classifier: beta(q) = {beta:.2f} for question: '{question}'")
        return beta

if __name__ == "__main__":
    qc = QuestionClassifier()
    print(qc.estimate_beta("what did the person say?"))
    print(qc.estimate_beta("what color is the car?"))
