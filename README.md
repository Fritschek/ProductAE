# ProductAE
- Implementation of the Product Autoencoder -

  This is an ongoing project to reproduce the results of the [publication](https://arxiv.org/pdf/2303.16424.pdf): `Jamali, M.V., Saber, H., Hatami, H., & Bae, J.H. (2021). ProductAE: Toward Training Larger Channel Codes based on Neural Product Codes. ICC 2022 - IEEE International Conference on Communications, 3898-3903`.
  Encoder and Decoder systems are implemented as in the publication, with some parameters still to fine-tune. Results are not as good as in the journal, which might be due to not using the very large batch size (50k). I'm limiting the batch size to max 2-3k for practical reasons and because I believe an AE should also work in that case.
  I would be happy about constructive comments to make the system work better with smaller batch sizes.
  

 
