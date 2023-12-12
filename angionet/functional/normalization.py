
def standardize(image, stats = None):
    if stats is not None:
        image = (image - stats[0]) / stats[1]
    else:
        image = (image - image.mean()) / image.std()

    return image


def rescale(image, stats = None):
    if stats is not None:
        image = (image - stats[0]) / (stats[1] - stats[0])
    else:
        image = (image - image.min()) / (image.max() - image.min())

    return image
