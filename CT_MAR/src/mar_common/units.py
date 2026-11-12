
def hu2mu(img, mu_water=0.02, huOffset=-1000.0):
    """HU -> mu (linear attenuation coefficient, per mm)."""
    return (img - huOffset) * (mu_water / 1000.0)


def mu2hu(img_mu, mu_water=0.02, huOffset=-1000.0):
    """mu (per mm) -> HU."""
    return img_mu * (1000 / mu_water) + huOffset