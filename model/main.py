from waste_classifier import WasteClassifier

classifier = WasteClassifier()
classifier.load_model('waste_classifier_model.pkl')

result = classifier.classify_image('testData/test1.jpg')
print(f"This is {result['predicted_class']} waste")
print(f"Put it in: {result['recommended_bin']}")
print(f"Confidence: {result['confidence']:.2%}")