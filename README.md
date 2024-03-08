# Audio-Classification
Developed a system to classify audio into different categories (e.g., speech, music, noise) using Mel-frequency cepstral coefficients (MFCCs) as features and a machine learning model (e.g., Random Forest).
Preprocessing Audio Data (Feature Extraction):

librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40): This line extracts Mel-frequency cepstral coefficients (MFCCs) from your audio signal (y). MFCCs are a common feature representation used in audio classification tasks because they capture the perceptual characteristics of sound.
np.mean(mfccs_features.T, axis=0): This line calculates the average of each feature across all time steps in the MFCCs. This is a dimensionality reduction technique that summarizes the overall spectral information of the audio.
mfccs_scaled_features.reshape(1, -1): This line reshapes the averaged features into a single row and appropriate number of columns, making it suitable for feeding into the machine learning model.
Classification (Assuming a Fitted Model):

fitted_model.predict(mfccs_scaled_features): This line assumes you have a trained machine learning model (fitted_model) that takes features as input and predicts class labels. It uses the preprocessed audio features (mfccs_scaled_features) to predict the class label for the audio.
Inverse Transformation (Optional):

labelencoder.inverse_transform(predicted_label): This line (if used) assumes you previously encoded class labels using a label encoder (labelencoder). This line converts the predicted numerical label back to its original human-readable format (e.g., "speech," "music," "noise").
Output:

print(predicted_class): This line prints the predicted class label for the audio sample.
