from waste_classifier import WasteClassifier

classifier = WasteClassifier(image_size=(64, 64))
history = classifier.train_model('data/')
classifier.save_model('waste_classifier_model.pkl')