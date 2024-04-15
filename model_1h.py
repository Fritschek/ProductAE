class OneHotEncoder(nn.Module):
    def __init__(self, M, h, n, **extra_kwargs):
        super(OneHotEncoder, self).__init__()
        
        layers = [nn.Linear(M, h), nn.SELU()]
        for _ in range(5):
            layers.extend([nn.Linear(h, h), nn.SELU()])
        layers.append(nn.Linear(h, n))

        self._f = nn.Sequential(*layers)
        
    def power_constraint(self, codes):
        codes_mean = torch.mean(codes)
        codes_std  = torch.std(codes)
        codes_norm = (codes - codes_mean) / codes_std
        return codes_norm
        
    def forward(self, inputs):
        x = self._f(inputs)
        return x


class OneHotDecoder(nn.Module):
    def __init__(self, M, h, n, **extra_kwargs):
        super(OneHotDecoder, self).__init__()
        
        layers = [nn.Linear(n, h), nn.SELU()]
        for _ in range(5):
            layers.extend([nn.Linear(h, h), nn.SELU()])
        layers.append(nn.Linear(h, M))

        self._f = nn.Sequential(*layers)
        
    def forward(self, inputs):
        x = self._f(inputs)
        return x


class OneHotDecoderLast(nn.Module):
    def __init__(self, M, h, n, **extra_kwargs):
        super(OneHotDecoderLast, self).__init__()
        
        layers = [nn.Linear(n, h), nn.SELU()]
        for _ in range(7):
            layers.extend([nn.Linear(h, h), nn.SELU()])
        layers.append(nn.Linear(h, M))

        self._f = nn.Sequential(*layers)
        
    def forward(self, inputs):
        x = self._f(inputs)
        return x


class OneHotProductAEEncoder(nn.Module):
    def __init__(self, M, N):
        super(OneHotProductAEEncoder, self).__init__()
        self.M = M
        self.n1, self.n2 = N
        self.encoders = nn.ModuleList()
        for m, n in zip(M, N):
            self.encoders.append(OneHotEncoder(m, 100, n))
            
    def forward(self, U):
        B = U.shape[0]
        U = U.reshape(B, self.M[0], self.M[1])
        
        # Step 1
        U1 = U.view(-1, self.encoders[0]._f[0].in_features)
        U1 = self.encoders[0](U1).view(U.shape[0], self.M[1], -1)
        
        # Step 2
        U2 = U1.transpose(1, 2).contiguous()
        U2 = U2.view(-1, self.encoders[1]._f[0].in_features)
        U2 = self.encoders[1](U2).view(U1.shape[0], self.n2, self.n1)
        
        # Step 3
        U2_flat = U2.view(U2.shape[0], -1)
        C = self.encoders[0].power_constraint(U2_flat)
        return C


class OneHotProdDecoder(nn.Module):
    def __init__(self, I, M, N, **extra_kwargs):
        super(OneHotProdDecoder, self).__init__()

        self.I = I
        self.M1, self.M2 = M
        self.n1, self.n2 = N
        self.F = 3
        H = 150

        self.decoders_1 = nn.ModuleList()
        self.decoders_2 = nn.ModuleList()

        for i in range(I):
            if i == 0:
                self.decoders_1.append(OneHotDecoder(self.F * self.n1, H, (1 + self.F) * self.n1, **extra_kwargs))
                self.decoders_2.append(OneHotDecoder(self.F * self.n2, H, self.n2, **extra_kwargs))
            elif i < I - 1:
                self.decoders_1.append(OneHotDecoder(self.F * self.n1, H, (1 + self.F) * self.n1, **extra_kwargs))
                self.decoders_2.append(OneHotDecoder(self.F * self.n2, H, (1 + self.F) * self.n2, **extra_kwargs))
            else:
                self.decoders_1.append(OneHotDecoderLast(self.M1, H, self.F * self.n1, **extra_kwargs))
                self.decoders_2.append(OneHotDecoderLast(self.F * self.M2, H, (1 + self.F) * self.n2, **extra_kwargs))

    def forward(self, Y):
        B = Y.size(0)
        Y = Y.view(B, self.n1, self.n2)
    
        if self.I == 1:
            Yin2_2 = Y
        else:
            for i in range(self.I-1):
                if i == 0:
                    Y2 = self.decoders_2[i](Y).view(B, self.F * self.n1, self.n2)
                else:
                    Yout2 = self.decoders_2[i](Yin2_2)
                    Y2 = (Yout2 - Yin2_1).view(B, self.F * self.n1, self.n2)

                Yin1 = torch.cat([Y, Y2], dim=1).permute(0, 2, 1)
                Y1 = self.decoders_1[i](Yin1).permute(0, 2, 1)
                Yin2_1 = (Y1 - Y2).reshape(B, self.n1, self.F * self.n2)
                Yin2_2 = torch.cat([Y, Yin2_1], dim=2)

        Y2 = self.decoders_2[-1](Yin2_2).view(B, self.F * self.n1, self.M2)
        Y1 = self.decoders_1[-1](Y2.permute(0, 2, 1))
        U_hat = Y1.view(B, self.M1 * self.M2)
        m = nn.Sigmoid()
        return m(U_hat)