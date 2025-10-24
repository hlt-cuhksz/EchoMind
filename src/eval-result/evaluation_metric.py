


# 1. Understanding
# WER
import evaluate              
def calculate_wer(references, predictions, warning_threshold=1):
    calculate_wer = evaluate.load("wer")
    wers = []
    for i in range(len(references)):
        wer = calculate_wer.compute(references=[references[i]], predictions=[predictions[i]])
        if wer > warning_threshold:
            print(f"High WER ({wer:.2f}) between reference: '{references[i]}' and prediction: '{predictions[i]}'")
        wers.append(wer)

    return wers


# SemSim 
from sentence_transformers import SentenceTransformer
import numpy as np

def calculate_embedding_and_cosine(model_name: str, text1: str, text2: str) -> dict:
    model = SentenceTransformer(model_name)
    
    embedding1 = model.encode([text1])
    embedding2 = model.encode([text2])
    
    cosine_similarity = float(np.dot(embedding1, embedding2.T)) 
    
    return {
        'embedding1': embedding1[0].tolist(),
        'embedding2': embedding2[0].tolist(),
        'cosine_similarity': cosine_similarity
    }


# 2.Response_text 
def calculate_bleu(self, reference: str, candidate: str) -> float:
    if not reference.strip() or not candidate.strip():
        return 0.0
        
    reference_tokens = nltk.word_tokenize(reference.lower())
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    
    smoothing_fn = getattr(self, "smoothing_function", SmoothingFunction().method1)
    
    bleu_score = sentence_bleu(
        [reference_tokens],
        candidate_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing_fn
    )
    return bleu_score

def calculate_rouge_l(self, reference: str, candidate: str) -> float:

    if not reference.strip() or not candidate.strip():
        return 0.0
        
    scores = self.rouge_scorer.score(reference, candidate)
    return scores['rougeL'].fmeasure

def calculate_meteor(self, reference: str, candidate: str) -> float:

    if not reference.strip() or not candidate.strip():
        return 0.0
        
    reference_tokens = reference.lower().split()
    candidate_tokens = candidate.lower().split()
    
    meteor_score_val = meteor_score([reference_tokens], candidate_tokens)
    return meteor_score_val
    
from bert_score import score
def calculate_bert_score(self, references, candidates,
                         model_type='bert-base-uncased',
                         lang='en', rescale_with_baseline=True):

    P, R, F1 = score(candidates, references,
                     model_type=model_type,
                     lang=lang,
                     rescale_with_baseline=rescale_with_baseline)
    return F1.mean().item()


# GPT evaluation
def construct_prompt(spoken_info: str, topic: str, speak_a: str, response_transcript: str) -> str:
    prompt_template = f"""
###TASK REQUIREMENTS:
You will be given a conversation context (including specific spoken_info, the conversation topic, and Speaker A's utterance) and a sentence that serves as a response to Speaker A's utterance in the INPUT section below.
Your task is to evaluate how well this response sentence performs across the following four dimensions by providing scores based on the four dimensions' criteria below. Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

1. Context Fit(1-5 point)
The score should reflect how well the response fits within the context of the scenario (i.e., topic, and speaker A's utterance). Focus on whether the response seems relevant to the conversation and addresses the elements in the case appropriately
1 point: The reply does not adapt to the dialogue background at all; it is unrelated to the topic or context and feels abrupt or unnatural.
2 points: The reply partially fits the dialogue background, but the content is not fully relevant and feels somewhat unnatural or lacks fluency.
3 points: The reply basically adapts to the dialogue background and is generally on-topic, but parts feel unnatural or slightly off-topic.
4 points: The reply adapts well to the dialogue background; the content is coherent and relevant, with minor room for improvement.
5 points: The reply fully matches the dialogue background; it is smooth and natural, perfectly fitting the context and situation.

2. Response Naturalness(1-5 point)
The score should reflect how naturally the response flows within the conversation. It considers whether the response sounds like something a real person would say in the given context.
1 point: The response feels stiff or robotic, lacking conversational fluency; it sounds like pre-written lines.
2 points: The response has some naturalness, but the tone or phrasing still feels slightly unnatural, with a rigid structure.
3 points: The response is generally natural, though somewhat formulaic; overall, it matches the rhythm and tone of everyday conversation.
4 points: The response is very natural, with a tone that fits casual dialogue; there are no noticeable awkward or unnatural elements.
5 points: The response is exceptionally natural, fully capturing the flow and authenticity of real conversation; it sounds like a genuine exchange between two people.

3. Colloquialism Degree(1-5 point)
Evaluate how informal or conversational the response content looks like. Checks if the response uses natural, everyday language, particularly in spoken or informal settings.
1 point: The response is entirely non-colloquial—overly formal or academic—and completely mismatched with everyday spoken language.
2 points: The response contains some colloquial elements, yet its overall tone remains fairly formal, lacking lived-in, natural phrasing.
3 points: The response strikes a moderate balance: it mixes formal and colloquial expressions, making it suitable for daily conversation but still slightly reserved.
4 points: The response is largely colloquial—warm, natural, and well-suited to informal exchanges, with only a trace of formality.
5 points: The response is fully colloquial, using the relaxed, authentic language of everyday dialogue; it feels effortless and natural.

4. Speech Information Relevance(1-5 point)
The score should evaluate how the response should be formulated based on the provided speech information {spoken_info}. The score should reflect how accurately the sentence addresses or incorporates the speech information {spoken_info} into this response.
1 point: The response is completely unrelated to the provided speech information {spoken_info}; it offers no content that reflects or addresses {spoken_info} in any way.
2 points: The response barely acknowledges the speech information {spoken_info} and instead presents content that is either contradictory or inconsistent with {spoken_info}.
3 points: The response somewhat overlooks the speech information {spoken_info}, failing to fully incorporate its characteristics, resulting in a reply that feels imprecise or biased.
4 points: The response takes the speech information {spoken_info} into account and shows some awareness of {spoken_info}, yet it does not fully integrate it into the conversation, making the reply somewhat stiff and leaving room for more natural expression.
5 points: The response is entirely grounded in the speech information {spoken_info}, accurately reflecting its relevant content and achieving a high degree of alignment with {spoken_info}.

Evaluation Steps:
1.Read the response sentence carefully and understand its relation to the context.
2.Analyze the sentence based on the criterias above.
3.Assign four scores that best represents how well the sentence fits the four dimensions, with 1 being the lowest and 5 being the highest.
4.Output the scores and the reasons for the scores for four dimensions in JSON key–value format.

##EVALUATION EXAMPLE：
"spoken_info": "male",
 "topic": "school",
 "speak_A": "Do you have any new books about space exploration?"

response to speaker A: "Of course! We just got some fascinating new books about space. Do you prefer the science-heavy ones, or are you more drawn to story-driven adventures?"
##EVALUATION EXAMPLE OUTPUT:

"context_fit_score": 4,
"context_fit_resaon": "The reply adapts to the context of asking about space books and offers appropriate follow-up questions. It is somewhat related to the school topic, but the choice of book categories could be further refined.",
"response_naturalness_score": 5,
"response_naturalness_resaon": "The reply sounds very natural. Asking Speaker A about their preferred book category is a realistic and appropriate response that fits the scenario and topic.",
"colloquialism_degree_score": 4,
"colloquialism_degree_resaon": "The reply maintains a high level of colloquialism, and the overall tone suits a conversational setting. Some word choices are slightly formal—appropriate for a school environment—but it still feels very friendly.",
"speech_information_relevance_score": 1,
"speech_information_relevance_reason": " Considering the voice information male, the reply does not include any content that references this information."


##INPUT:
"spoken_info": "{spoken_info}",
"topic": "{topic}",
"speak_A": "{speak_a}",
"response_transcript": "{response_transcript}"
"""
    
    return prompt_template

# 3.Response_audio
# Gemini evaluation
system_prompt = """
You are a professional speech emotion and style evaluator. 
Your task is to evaluate the **Vocal Empathy Score (VES)** of a response speech.

###Definition:
Vocal Empathy Score measures how well the responder's speech expresses an appropriate emotional tone and vocal style to match the speaker's described state.
- Ignore content semantic accuracy.
- Focus on emotional resonance, tone, vocal delivery, and non-verbal vocal cues.
- Style cues may include: emotional tone, pitch contour, speech rate, volume, pauses, timbre, and non-verbal sounds (laughter, cough, sigh).

###Scoring scale:
5 = Perfect empathy: The responder's vocal emotional intensity, pitch, rhythm, and tone highly match the speaker's state, conveying appropriate care or emotional resonance. Example: Speaker: Low, hoarse, coughing → Responder: Gentle, slower pace, lower volume, with a concerned tone.
4 = Basic empathy: The vocal style of the responder generally matches the speaker's state, but there are minor deficiencies, such as the emotional intensity being slightly weaker or missing subtle pauses. Example: Speaker: Tired → Responder: Soft volume but relatively fast pace.
3 = Weak empathy: The direction is correct, with some resonance, but the emotional expression is insufficient or lacks key vocal features. Example: Speaker: Excited → Responder: Mostly flat tone, with slightly higher volume on a few words.
2 = Incorrect empathy: Most of the style doesn't match the speaker's state, even opposite to it. Example: Speaker: Depressed → Responder: Lively, cheerful high pitch.
1 = No empathy: The vocal style shows no emotional expression at all, sounding mechanical and monotonous. Example: Speaker: Tired → Neutral tone, no emotional variation, and rigid tone.


###Response format:
Return your answer in JSON:
{
"VES_score": integer between 1 and 5,
"explanation": "<brief explanation>"
}

"""

# NISQA
class NISQAEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
    
    def evaluate(self, audio_file, **kwargs):
        args = {
            'mode': 'predict_file',
            'pretrained_model': 'NISQA/weights/nisqa.tar',
            'deg': audio_file,
            'output_dir': './tmp',
            'data_dir': None,
            'csv_file': None,
            'csv_deg': None,
            'num_workers': 0,
            'bs': 1,
            'ms_channel': None,
            'ms_max_segments': 10000,
        }
        self.model = nisqaModel(args)
        
        results = self.model.predict().to_dict()
        return results

# DNSMOS
class DNSMOSEvaluator(BaseEvaluator):
    def __init__(self, personalized_MOS=False):
        super().__init__()
        p808_model_path = 'DNSMOS/DNSMOS/model_v8.onnx'
        if personalized_MOS:
            primary_model_path = 'DNSMOS/pDNSMOS/sig_bak_ovr.onnx'
        else:
            primary_model_path = 'DNSMOS/DNSMOS/sig_bak_ovr.onnx'
            
        self.compute_score = ComputeScore(primary_model_path, p808_model_path)
        self.personalized_MOS = personalized_MOS

    def evaluate(self, audio_file, **kwargs):
        rec_result = self.compute_score(audio_file, SAMPLING_RATE, is_personalized_MOS=self.personalized_MOS)
        for k, v in rec_result.items():
            if isinstance(v, np.float32) or isinstance(v, np.float64):
                rec_result[k] = float(v)
        return rec_result

# emotion2vec
class Emotion2VecEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        self.model_id = "iic/emotion2vec_plus_large"
        self.model = AutoModel(
            model=self.model_id,
            hub="ms",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
        )

    @on_exception(expo, Exception, max_tries=3)
    def evaluate(self, audio_file, **kwargs):
        rec_result = self.model.generate(audio_file, output_dir="./tmp", granularity="utterance", extract_embedding=False)
        return rec_result

