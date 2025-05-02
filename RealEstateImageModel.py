import logging
import numpy as np
import os
from PIL import Image
import requests
from io import BytesIO
import time
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from label_studio_ml.model import LabelStudioMLBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants
LABEL_STUDIO_DATA_DIR = '/label-studio/data'
MODEL_PATH = os.getenv('MODEL_PATH', '/path/to/your/model.h5')  # Use environment variable or default path

class RealEstateImageModel(LabelStudioMLBase):
    def __init__(self, **kwargs):
        """Initialize the model with classification categories and check TensorFlow availability."""
        super(RealEstateImageModel, self).__init__(**kwargs)
        self.model = None
        self.initialized = False
        self.dummy_mode = not tf.__version__.startswith('2.')
        logger.info(f"Initialized, TensorFlow available: {not self.dummy_mode}")

        # Classification categories
        self.room_type_choices = ['kitchen', 'living room', 'bedroom', 'bathroom', 'dining room', 'office', 'basement', 'garage', 'outdoor', 'hallway', 'laundry room', 'other']
        self.quality_of_finishes_choices = ['high', 'mid-high', 'mid', 'mid-low', 'low']
        self.recency_of_renovation_choices = ['recent', 'modern', 'dated', 'original']
        self.lighting_choices = ['abundant natural', 'well-lit artificial', 'sufficiently-lit natural', 'dim']
        self.layout_choices = ['open plan', 'traditional', 'segmented', 'spacious', 'cramped']
        self.condition_choices = ['move-in ready', 'needs minor repairs', 'needs major renovation']
        self.desirable_feature_choices = ['present', 'absent']
        self.undesirable_feature_choices = ['present', 'absent']
        self.room_size_choices = ['small', 'medium', 'large']
        self.room_functionality_choices = ['functional', 'decorative', 'multi-purpose']
        self.appliance_quality_choices = ['high-end', 'standard', 'outdated']

    def setup(self, **kwargs):
        """Set up the model by loading pre-trained weights or switching to dummy mode on failure."""
        logger.info("Setting up model")
        try:
            if self.model is None:
                if self.dummy_mode:
                    logger.info("Using dummy mode due to TensorFlow version incompatibility")
                    self.model = "dummy"
                else:
                    # Load pre-trained model from file
                    self.model = load_model(MODEL_PATH)
                    logger.info(f"Pre-trained model loaded from {MODEL_PATH}")
            self.initialized = True
            logger.info("Model setup complete")
            return self
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            self.dummy_mode = True
            self.model = "dummy"
            self.initialized = True
            logger.info("Switched to dummy mode due to setup failure")
            return self

    def predict(self, tasks, **kwargs):
        """Generate predictions for a list of tasks."""
        logger.info(f"Predicting for {len(tasks)} tasks")
        if not self.initialized:
            self.setup()

        predictions = []
        for task in tasks:
            try:
                image_url = task['data'].get('image')
                if not image_url:
                    logger.warning(f"No image URL found in task: {task}")
                    predictions.append({'result': self._format_dummy_predictions()})
                    continue

                logger.info(f"Processing image: {image_url}")
                image = self._download_image(image_url)
                image = self._preprocess_image(image)
                if self.dummy_mode:
                    result = self._format_dummy_predictions()
                else:
                    preds = self.model.predict(image, verbose=0)
                    result = self._format_predictions(preds)
                predictions.append({'result': result})
            except Exception as e:
                logger.error(f"Prediction failed for task {task}: {str(e)}")
                predictions.append({'result': self._format_dummy_predictions()})
        return predictions

    def _download_image(self, url):
        """Download an image from a local path or remote URL with retry logic."""
        if url.startswith('/data/upload/'):
            relative_path = url[len('/data/'):]
            full_path = os.path.join(LABEL_STUDIO_DATA_DIR, relative_path)
            logger.info(f"Loading local image: {full_path}")
            if not os.path.exists(full_path):
                logger.error(f"File not found: {full_path}")
                raise Exception(f"Local file not found: {url}")
            image = Image.open(full_path).convert('RGB')
            return image
        else:
            for attempt in range(3):
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                    logger.info(f"Loaded web image: {url}")
                    return image
                except Exception as e:
                    logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
                    if attempt == 2:
                        raise Exception(f"Failed to download image {url}: {str(e)}")
                    time.sleep(2)

    def _preprocess_image(self, image):
        """Preprocess the image for model input."""
        image = image.resize((224, 224))
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        return image

    def _format_predictions(self, preds):
        """Format model predictions into Label Studio-compatible output."""
        room_type_pred = preds[0][0]
        quality_of_finishes_pred = preds[1][0]
        recency_of_renovation_pred = preds[2][0]
        lighting_pred = preds[3][0]
        layout_pred = preds[4][0]
        condition_pred = preds[5][0]
        desirable_feature_pred = preds[6][0]
        undesirable_feature_pred = preds[7][0]
        room_size_pred = preds[8][0]
        room_functionality_pred = preds[9][0]
        appliance_quality_pred = preds[10][0]

        result = [
            {
                'from_name': 'room_type',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': [self.room_type_choices[np.argmax(room_type_pred)]]}
            },
            {
                'from_name': 'quality_of_finishes',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': [self.quality_of_finishes_choices[np.argmax(quality_of_finishes_pred)]]}
            },
            {
                'from_name': 'recency_of_renovation',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': [self.recency_of_renovation_choices[np.argmax(recency_of_renovation_pred)]]}
            },
            {
                'from_name': 'lighting',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': [self.lighting_choices[np.argmax(lighting_pred)]]}
            },
            {
                'from_name': 'layout',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': [self.layout_choices[np.argmax(layout_pred)]]}
            },
            {
                'from_name': 'condition',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': [self.condition_choices[np.argmax(condition_pred)]]}
            },
            {
                'from_name': 'desirable_feature',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': [self.desirable_feature_choices[np.argmax(desirable_feature_pred)]]}
            },
            {
                'from_name': 'undesirable_feature',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': [self.undesirable_feature_choices[np.argmax(undesirable_feature_pred)]]}
            },
            {
                'from_name': 'room_size',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': [self.room_size_choices[np.argmax(room_size_pred)]]}
            },
            {
                'from_name': 'room_functionality',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': [self.room_functionality_choices[np.argmax(room_functionality_pred)]]}
            },
            {
                'from_name': 'appliance_quality',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': [self.appliance_quality_choices[np.argmax(appliance_quality_pred)]]}
            }
        ]
        return result

    def _format_dummy_predictions(self):
        """Generate random dummy predictions for testing or fallback."""
        logger.info("Generating dummy predictions")
        return [
            {
                'from_name': 'room_type',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': [np.random.choice(self.room_type_choices)]}
            },
            {
                'from_name': 'quality_of_finishes',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': [np.random.choice(self.quality_of_finishes_choices)]}
            },
            {
                'from_name': 'recency_of_renovation',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': [np.random.choice(self.recency_of_renovation_choices)]}
            },
            {
                'from_name': 'lighting',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': [np.random.choice(self.lighting_choices)]}
            },
            {
                'from_name': 'layout',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': [np.random.choice(self.layout_choices)]}
            },
            {
                'from_name': 'condition',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': [np.random.choice(self.condition_choices)]}
            },
            {
                'from_name': 'desirable_feature',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': [np.random.choice(self.desirable_feature_choices)]}
            },
            {
                'from_name': 'undesirable_feature',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': [np.random.choice(self.undesirable_feature_choices)]}
            },
            {
                'from_name': 'room_size',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': [np.random.choice(self.room_size_choices)]}
            },
            {
                'from_name': 'room_functionality',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': [np.random.choice(self.room_functionality_choices)]}
            },
            {
                'from_name': 'appliance_quality',
                'to_name': 'image',
                'type': 'choices',
                'value': {'choices': [np.random.choice(self.appliance_quality_choices)]}
            }
        ]