import time
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from nltk.translate.bleu_score import SmoothingFunction


# Load sentence transformer model for evaluation
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load sentiment analysis model
roberta_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
sentiment_pipeline = pipeline("text-classification", model=roberta_model_name)

# =====================
# 1️⃣ Sentiment Analysis Function
# =====================

def analyze_sentiment(text):
    """Predict sentiment using nlptown/bert-base-multilingual-uncased-sentiment"""
    result = sentiment_pipeline(text)[0]
    sentiment_map = {
        "1 star": "Negative",
        "2 stars": "Negative",
        "3 stars": "Neutral",
        "4 stars": "Positive",
        "5 stars": "Positive"
    }
    return sentiment_map.get(result["label"], "Neutral")

# =====================
# 2️⃣ RAG Evaluation Functions
# =====================

def precision_at_k(retrieved_docs, relevant_docs, k):
    """Computes Precision@K."""
    retrieved_at_k = retrieved_docs[:k]
    relevant_retrieved = sum(1 for doc in retrieved_at_k if doc in relevant_docs)
    return relevant_retrieved / k

def recall_at_k(retrieved_docs, relevant_docs, k):
    """Computes Recall@K."""
    retrieved_at_k = retrieved_docs[:k]
    relevant_retrieved = sum(1 for doc in retrieved_at_k if doc in relevant_docs)
    return relevant_retrieved / len(relevant_docs)

def mean_reciprocal_rank(retrieved_docs_list, relevant_docs_list):
    """Computes Mean Reciprocal Rank (MRR)."""
    mrr_total = 0
    for retrieved_docs, relevant_docs in zip(retrieved_docs_list, relevant_docs_list):
        for rank, doc in enumerate(retrieved_docs, start=1):
            if doc in relevant_docs:
                mrr_total += 1 / rank
                break
    return mrr_total / len(retrieved_docs_list)

def calculate_bleu(reference, generated):
    """Computes BLEU Score with smoothing."""
    smoothie = SmoothingFunction().method1  # Smoothing for short sentences
    return sentence_bleu([reference.split()], generated.split(), smoothing_function=smoothie)


def calculate_rouge(reference, generated):
    """Computes ROUGE Score."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores["rougeL"].fmeasure

def calculate_meteor(reference, generated):
    """Computes METEOR Score with tokenized inputs."""
    reference_tokens = reference.split()  # Tokenize reference
    generated_tokens = generated.split()  # Tokenize generated response
    return meteor_score([reference_tokens], generated_tokens)

def context_recall(retrieved_context, generated_response):
    """Computes Context Recall."""
    retrieved_words = set(retrieved_context.split())
    generated_words = set(generated_response.split())
    return len(retrieved_words & generated_words) / len(retrieved_words)

def answer_faithfulness(retrieved_context, generated_response):
    """Computes Faithfulness Score using Cosine Similarity."""
    context_embedding = embedding_model.encode(retrieved_context, convert_to_tensor=True)
    response_embedding = embedding_model.encode(generated_response, convert_to_tensor=True)
    return util.pytorch_cos_sim(context_embedding, response_embedding).item()

def measure_latency(function, *args):
    """Measures function execution time."""
    start_time = time.time()
    function(*args)
    return time.time() - start_time

# =====================
# 3️⃣ Running Evaluation on Sample Data
# =====================

retrieved_docs = ["Like PMR, the more you regularly practice guided imagery, especially for ten minutes or more, the better able you are to cue yourself with the same imagery to create relaxation and peace during stressful times. It is best to practice in a quiet setting, where you are comfortable and not easily disturbed. Put your phone away and put a DO NOT DISTURB sign on your door if you have to. You may choose your office, bedroom, parked car, bathtub, or create a meditation area in your home. The more you rehearse when you are not stressed, the easier it will be to use this imagery when you are in a bad mood or triggered. Exercise 5.6: Happy Place Visualization • Close your eyes. • Take note of how you feel. You may rate your negative emotional intensity at the time from zero to ten (ten being very upset, angry, sad, and so forth, and zero being you feel no negative emotions). Take note of how your body feels as well. • Do five or six power breaths, deciding to let go of your emotions and thoughts every time you exhale. • Bring your thoughts and imagination to something positive. For example, imagine yourself in your favorite place in nature—perhaps relaxing on a beach, walking through a forest, hiking on a mountain, lying in a hammock, working in your garden, watching the sunset from a boat—or in a place where you are always happy and relaxed. • Take note of all the details: What are you feeling? How are you feeling? What are you wearing? Who are you with? How does the air feel on your skin? What colors do you notice—the blueness of the water, colors of the sky in a sunrise, the rosy cheeks of someone you love? What sounds do you hear—waves crashing against the shore, birds singing, or leaves moving in the breeze? Do you taste or smell anything—salt on your lips, chocolate or the aroma of wood burning in a fireplace? The more details you can find and the more senses you evoke, the better, as this memory engages your concentration in a direction that is opposite to your negative thinking and anger. • Wherever you find yourself, imagine yourself feeling happy and at peace and smiling. Try to stay with this imagery, exploring all of your senses for at least five to ten minutes. • Take note of how you feel now, rating your negative emotional intensity from zero to ten. • If you feel sufficiently relaxed and when you are ready, do another five or six power breaths. • Remind yourself that you can go to this wonderful place any time. It is yours to go to. • Now open your eyes. As I mentioned, when you quiet the mind, relax the body and breathe deeply for an extended period of time, you meditate. When you add in visual imagery intended to not only create calm but also address the stress and underlying negative beliefs, you can heal on a deeper level as well. You can use imagery to explore your negative emotions and where they originate from and to imagine new situations that dissolve the negative feelings. In addition, when you combine peaceful and loving visual imagery with physical exercises, like PMR, you can train your mind and body to associate certain peaceful images with the relaxation of the muscles. Again, the more you practice this technique, the more your body learns how to be cued to let go and relax. Exercise 5.7: A Meditation Combo Whether you choose to do this meditation in the moment for a few minutes when you are upset or as part of your meditation practice, you will benefit, as it combines PMR with imagery in a way that can help you dissolve your stress and find bliss. • Close your eyes. • Take note of how you feel and rate your negative emotional intensity level. • Do five or six power breaths. • Imagine the sun is shining down upon you, golden rays of healing light full of love and wisdom. • Tense your forehead, squeezing the muscles of your forehead by raising your eyebrows as high as you can, then hold your breath for about five seconds. • Exhale slowly as you imagine the golden light shining down on the top of your head and now moving down your forehead, causing the muscles of your forehead to relax and release all the tension. • Tense your mouth, squeezing the jaw muscles, smiling as wide as you can, then hold your breath for about five seconds. • Exhale slowly as you imagine the golden light shining down through the muscles of your face as they relax and release all the tension. • Tense your neck and shoulders, squeezing the shoulder up to your ears, then hold your breath for about five seconds. • Exhale slowly as you imagine the golden light shining down through the muscles of your neck and shoulders as they relax and release all the tension. • Tense your chest and abdomen, squeezing the muscles of your abdomen and holding your breath for about five seconds. • Exhale slowly as you imagine the golden light shining down through the muscles of your chest and abdomen as they relax and release all the tension. • Tense your buttocks, squeezing the buttocks muscles, and hold your breath for about five seconds. • Exhale slowly as you imagine the golden light shining down through the muscles of your buttocks as they relax and release all the tension. • Tense both your arms and hands, making fists as you flex your muscles, then hold your breath for about five seconds. • Exhale slowly as you imagine the golden light shining down through the muscles of both your arms and hands as they relax and release all the tension. • Tense both your legs and feet, squeezing your thighs and curling your toes under, and hold your breath for about five seconds. • Exhale slowly as you imagine the golden light shining down through the muscles of both your legs and feet as they relax and release all the tension. • You are filled with and surrounded by golden light."]
relevant_docs = ["Like PMR, the more you regularly practice guided imagery, especially for ten minutes or more, the better able you are to cue yourself with the same imagery to create relaxation and peace during stressful times. It is best to practice in a quiet setting, where you are comfortable and not easily disturbed. Put your phone away and put a DO NOT DISTURB sign on your door if you have to. You may choose your office, bedroom, parked car, bathtub, or create a meditation area in your home. The more you rehearse when you are not stressed, the easier it will be to use this imagery when you are in a bad mood or triggered. Exercise 5.6: Happy Place Visualization • Close your eyes. • Take note of how you feel. You may rate your negative emotional intensity at the time from zero to ten (ten being very upset, angry, sad, and so forth, and zero being you feel no negative emotions). Take note of how your body feels as well. • Do five or six power breaths, deciding to let go of your emotions and thoughts every time you exhale. • Bring your thoughts and imagination to something positive. For example, imagine yourself in your favorite place in nature—perhaps relaxing on a beach, walking through a forest, hiking on a mountain, lying in a hammock, working in your garden, watching the sunset from a boat—or in a place where you are always happy and relaxed. • Take note of all the details: What are you feeling? How are you feeling? What are you wearing? Who are you with? How does the air feel on your skin? What colors do you notice—the blueness of the water, colors of the sky in a sunrise, the rosy cheeks of someone you love? What sounds do you hear—waves crashing against the shore, birds singing, or leaves moving in the breeze? Do you taste or smell anything—salt on your lips, chocolate or the aroma of wood burning in a fireplace? The more details you can find and the more senses you evoke, the better, as this memory engages your concentration in a direction that is opposite to your negative thinking and anger. • Wherever you find yourself, imagine yourself feeling happy and at peace and smiling. Try to stay with this imagery, exploring all of your senses for at least five to ten minutes. • Take note of how you feel now, rating your negative emotional intensity from zero to ten. • If you feel sufficiently relaxed and when you are ready, do another five or six power breaths. • Remind yourself that you can go to this wonderful place any time. It is yours to go to. • Now open your eyes. As I mentioned, when you quiet the mind, relax the body and breathe deeply for an extended period of time, you meditate. When you add in visual imagery intended to not only create calm but also address the stress and underlying negative beliefs, you can heal on a deeper level as well. You can use imagery to explore your negative emotions and where they originate from and to imagine new situations that dissolve the negative feelings. In addition, when you combine peaceful and loving visual imagery with physical exercises, like PMR, you can train your mind and body to associate certain peaceful images with the relaxation of the muscles. Again, the more you practice this technique, the more your body learns how to be cued to let go and relax. Exercise 5.7: A Meditation Combo Whether you choose to do this meditation in the moment for a few minutes when you are upset or as part of your meditation practice, you will benefit, as it combines PMR with imagery in a way that can help you dissolve your stress and find bliss. • Close your eyes. • Take note of how you feel and rate your negative emotional intensity level. • Do five or six power breaths. • Imagine the sun is shining down upon you, golden rays of healing light full of love and wisdom. • Tense your forehead, squeezing the muscles of your forehead by raising your eyebrows as high as you can, then hold your breath for about five seconds. • Exhale slowly as you imagine the golden light shining down on the top of your head and now moving down your forehead, causing the muscles of your forehead to relax and release all the tension. • Tense your mouth, squeezing the jaw muscles, smiling as wide as you can, then hold your breath for about five seconds. • Exhale slowly as you imagine the golden light shining down through the muscles of your face as they relax and release all the tension. • Tense your neck and shoulders, squeezing the shoulder up to your ears, then hold your breath for about five seconds. • Exhale slowly as you imagine the golden light shining down through the muscles of your neck and shoulders as they relax and release all the tension. • Tense your chest and abdomen, squeezing the muscles of your abdomen and holding your breath for about five seconds. • Exhale slowly as you imagine the golden light shining down through the muscles of your chest and abdomen as they relax and release all the tension. • Tense your buttocks, squeezing the buttocks muscles, and hold your breath for about five seconds. • Exhale slowly as you imagine the golden light shining down through the muscles of your buttocks as they relax and release all the tension. • Tense both your arms and hands, making fists as you flex your muscles, then hold your breath for about five seconds. • Exhale slowly as you imagine the golden light shining down through the muscles of both your arms and hands as they relax and release all the tension. • Tense both your legs and feet, squeezing your thighs and curling your toes under, and hold your breath for about five seconds. • Exhale slowly as you imagine the golden light shining down through the muscles of both your legs and feet as they relax and release all the tension. • You are filled with and surrounded by golden light."]

generated_response = "Reducing stress after work can be beneficial for both your mental and physical well-being. Here are several strategies you might consider:\n\n1. **Physical Activity**: Engaging in some form of exercise, whether it’s a workout, a brisk walk, or yoga, can help release endorphins and improve your mood.\n\n2. **Mindfulness and Meditation**: Take a few minutes to practice mindfulness or meditation. Even short sessions can help calm your mind and reduce anxiety.\n\n3. **Hobbies and Interests**: Pursue activities you enjoy, like reading, drawing, or gardening. Engaging in hobbies can provide a positive distraction.\n\n4. **Socializing**: Spend time with friends or family. Social connections can be a strong buffer against stress.\n\n5. **Nature Exposure**: If possible, spend time outdoors. Nature can have a soothing effect and help clear your mind.\n\n6. **Breathing Exercises**: Practicing deep breathing can help activate the body's relaxation response, reducing stress effectively.\n\n7. **Limit Screen Time**: Reducing exposure to screens, especially before bed, can help your mind unwind.\n\n8. **Establish a Routine**: Create a post-work routine that includes relaxing activities. This can signal to your body that it's time to unwind.\n\n9. **Professional Help**: If stress becomes overwhelming, consider speaking with a therapist or counselor for guidance.\n\n10. **Healthy Eating and Hydration**: Maintaining good nutrition and staying hydrated can improve your overall mood and energy levels.\n\nExperiment with these strategies to see which ones work best for you. Everyone manages stress differently, so it may take some time to find your ideal approach."
reference_response = "Reducing stress after work is important for maintaining mental well-being. Here are some effective ways to unwind and relax:\n 1. Physical Activities - Exercise: Go for a walk, do yoga, or hit the gym to release endorphins. Stretching: Relieves tension built up from sitting all day. Deep Breathing: Try the 4-7-8 breathing technique to calm your nervous system.\n 2. Mental Relaxation - Meditation & Mindfulness: Helps clear your mind and reduce stress. Listening to Music: Calming or favorite tunes can uplift your mood. Reading a Book: Escaping into a good book can be very relaxing.\n 3. Engaging in Hobbies - Creative Outlets: Painting, writing, or playing an instrument. Gardening: Connecting with nature is a great stress reliever. Cooking: Trying a new recipe can be a fun distraction.\n 4. Social Interaction - Talk to Friends/Family: Venting can lighten the emotional load. Join a Club or Group: Engaging in social activities can boost happiness.\n 5. Digital Detox - Limit Screen Time: Avoid work emails and social media before bed. Watch a Feel-Good Movie: Comedy or light-hearted shows help relax your mind.\n 6. Self-Care - Take a Warm Bath: Helps relax your muscles and mind. Aromatherapy: Essential oils like lavender can reduce anxiety. Massage or Spa Treatment: Releases tension from your body.\n 7. Sleep & Rest - Follow a Sleep Routine: Aim for 7-9 hours of sleep. Avoid Caffeine Late in the Day: Helps ensure deep sleep. Use White Noise or Meditation Apps: Can improve sleep quality.\n Would you like recommendations based on your daily routine? 😊"

k =10  # Top K for retrieval metrics

metrics = {

    "Recall@K": recall_at_k(retrieved_docs, relevant_docs, k),
    "MRR": mean_reciprocal_rank([retrieved_docs], [relevant_docs]),
    "Answer Faithfulness": answer_faithfulness(" ".join(retrieved_docs), generated_response),
    "Generation Latency (s)": measure_latency(calculate_bleu, reference_response, generated_response)
}

# Print Results
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# =====================
# 4️⃣ Running Sentiment Analysis on CSV
# =====================

df = pd.read_csv("testdata.csv")

def map_sentiment_label(label):
    """Convert star ratings into sentiment categories"""
    mapping = {
        "1 star": "negative",
        "2 stars": "negative",
        "3 stars": "neutral",
        "4 stars": "positive",
        "5 stars": "positive"
    }
    return mapping.get(label, "neutral")

df["predicted_sentiment"] = df["query"].apply(lambda x: map_sentiment_label(analyze_sentiment(x)))
df["correct"] = df["predicted_sentiment"] == df["actual_sentiment"]
accuracy = df["correct"].mean() * 100

#print(f"Sentiment Analysis Accuracy: {accuracy:.2f}%")
print(df[["query", "actual_sentiment", "predicted_sentiment", "correct"]])
df.to_csv("sentiment_analysis_results.csv", index=False)
