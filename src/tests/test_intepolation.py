class TestInterpolation:
    def test_bilinear_interp(self):
        # get the image dimensions
        height, width = image_data.shape
        # get the resize factor
        factor = self.factor
        # calculate 
