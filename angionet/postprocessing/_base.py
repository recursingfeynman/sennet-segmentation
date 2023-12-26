class BaseTransform:
    def augment(self, image):
        raise NotImplementedError("Augmentation method not implemented.")

    def disaugment(self, image):
        raise NotImplementedError("Inverse method not implemented.")
