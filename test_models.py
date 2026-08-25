import traceback
try:
    import face_recognition_models
    print(face_recognition_models.pose_predictor_model_location())
except Exception as e:
    traceback.print_exc()
