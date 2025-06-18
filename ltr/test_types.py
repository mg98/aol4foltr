import unittest
import numpy as np
from ltr.types import ClickThroughRecord, FeatureVector, TermBasedFeatures, Statistics


class TestClickThroughRecord(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures with a sample ClickThroughRecord."""
        # Create a sample Statistics object
        sample_stats = Statistics()
        sample_stats.min = 1.0
        sample_stats.max = 5.0
        sample_stats.sum = 15.0
        sample_stats.mean = 3.0
        sample_stats.variance = 2.0
        
        # Create a sample TermBasedFeatures object
        sample_term_features = TermBasedFeatures()
        sample_term_features.bm25 = 2.5
        sample_term_features.tf = sample_stats
        sample_term_features.idf = sample_stats
        sample_term_features.tf_idf = sample_stats
        sample_term_features.stream_length = sample_stats
        sample_term_features.stream_length_normalized_tf = sample_stats
        sample_term_features.cos_sim = 0.8
        sample_term_features.covered_query_term_number = 3
        sample_term_features.covered_query_term_ratio = 0.75
        sample_term_features.char_len = 100
        sample_term_features.term_len = 20
        sample_term_features.total_query_terms = 4
        sample_term_features.exact_match = 1
        sample_term_features.match_ratio = 0.9
        
        # Create a sample FeatureVector
        sample_feature_vector = FeatureVector()
        sample_feature_vector.title = sample_term_features
        sample_feature_vector.body = sample_term_features
        sample_feature_vector.url = sample_term_features
        sample_feature_vector.number_of_slash_in_url = 3
        
        # Create the ClickThroughRecord
        self.original_record = ClickThroughRecord(
            rel=True,
            qid=12345,
            feat=sample_feature_vector
        )
    
    def test_to_array_from_array_round_trip(self):
        """Test that converting to array and back preserves the original data."""
        # Convert to array
        array_representation = self.original_record.to_array()
        
        # Verify array is numpy array
        self.assertIsInstance(array_representation, np.ndarray)
        
        # Convert back from array
        reconstructed_record = ClickThroughRecord.from_array(array_representation)
        
        # Assert the basic attributes are preserved
        self.assertEqual(self.original_record.rel, reconstructed_record.rel)
        self.assertEqual(self.original_record.qid, reconstructed_record.qid)
        
        # Assert that the feature vectors have the same values
        original_features = self.original_record.feat.features
        reconstructed_features = reconstructed_record.feat.features
        
        # Use numpy.allclose for floating point comparison
        np.testing.assert_allclose(
            original_features, 
            reconstructed_features,
            rtol=1e-7,
            atol=1e-7,
            err_msg="Feature vectors do not match after round-trip conversion"
        )
    
    def test_array_shape_and_dtype(self):
        """Test that the array has expected shape and data type."""
        array_representation = self.original_record.to_array()
        
        # Check data type
        self.assertEqual(array_representation.dtype, np.float32)
        
        # Check that array length matches expected size (rel + qid + features)
        expected_length = 2 + FeatureVector.n_features()  # rel + qid + feature count
        self.assertEqual(len(array_representation), expected_length)
    
    def test_array_values_correctness(self):
        """Test that the array contains the correct values in the right positions."""
        array_representation = self.original_record.to_array()
        
        # Check rel (index 0)
        self.assertEqual(bool(array_representation[0]), self.original_record.rel)
        
        # Check qid (index 1)
        self.assertEqual(int(array_representation[1]), self.original_record.qid)
        
        # Check features (index 2 onwards)
        expected_features = self.original_record.feat.features
        actual_features = array_representation[2:].tolist()
        
        np.testing.assert_allclose(
            expected_features,
            actual_features,
            rtol=1e-7,
            atol=1e-7,
            err_msg="Features in array do not match original features"
        )
    
    def test_from_array_with_simple_values(self):
        """Test from_array with a simple, manually created array."""
        # Create a simple array with known values
        n_features = FeatureVector.n_features()
        simple_array = np.zeros(2 + n_features, dtype=np.float32)
        simple_array[0] = 1.0  # rel = True
        simple_array[1] = 999  # qid = 999
        # Leave features as zeros for simplicity
        
        # Create record from array
        record = ClickThroughRecord.from_array(simple_array)
        
        # Verify basic attributes
        self.assertTrue(record.rel)
        self.assertEqual(record.qid, 999)
        self.assertIsInstance(record.feat, FeatureVector)
        
        # Verify features are all zeros (as expected)
        features = record.feat.features
        self.assertTrue(all(f == 0.0 for f in features))


if __name__ == '__main__':
    unittest.main() 