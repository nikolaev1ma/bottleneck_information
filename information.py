import numpy as np
import probability
from collections import Counter
import json

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


class Information:
    def __init__(self, net, dataset):
        self.net = net
        self.dataset = dataset

    def get_X_Y(self):
        X_array = []
        Y_array = []
        for i, (images, labels) in enumerate(self.dataset):
            X_array.append(images.cpu().detach().numpy())
            Y_array.append(labels.cpu().detach().numpy())
        self.X = np.concatenate(X_array)
        self.Y = np.concatenate(Y_array)
        self.n_samples = len(self.X)

    def get_data(self, net):
        Z1_array = []
        Z2_array = []
        Z3_array = []
        X_array = []
        Y_array = []
        for i, (images, labels) in enumerate(self.dataset):
            z1, z2, z3 = net.forwards_output(images)
            Z1_array.append(z1.cpu().detach().numpy())
            Z2_array.append(z2.cpu().detach().numpy())
            Z3_array.append(z3.cpu().detach().numpy())
            X_array.append(images.cpu().detach().numpy())
            Y_array.append((labels.cpu().detach().numpy()))
        self.Z1 = np.concatenate(Z1_array)
        self.Z2 = np.concatenate(Z2_array)
        self.Z3 = np.concatenate(Z3_array)
        self.X = np.concatenate(X_array)
        self.Y = np.concatenate(Y_array)


    def get_data_comp(self, net, version):
        Z1_array = []
        Z2_array = []
        Z3_array = []
        X_array = []
        Y_array = []
        for i, (images, labels) in enumerate(self.dataset):
            x, z1, z2, z3 = net.forward_compressed(images, version)
            X_array.append(x.cpu().detach().numpy())
            Z1_array.append(z1.cpu().detach().numpy())
            Z2_array.append(z2.cpu().detach().numpy())
            Z3_array.append(z3.cpu().detach().numpy())
            Y_array.append((labels.cpu().detach().numpy()))
        self.Z1 = np.concatenate(Z1_array)
        self.Z2 = np.concatenate(Z2_array)
        self.Z3 = np.concatenate(Z3_array)
        self.X = np.concatenate(X_array)
        self.Y = np.concatenate(Y_array)

    def discrete_binary(self, hidden):
        X_mins = 0.
        X_maxs = 0.5
        bins = np.linspace(X_mins, X_maxs, num=2)
        indexes = np.digitize(hidden, bins)
        return indexes.tolist()

    def discrete_hidden(self, hidden, num_count):
        hidden = hidden.reshape(hidden.shape[0], -1)
        hidden_mins = np.quantile(hidden, 0.01, axis=0) - 0.1
        hidden_maxs = np.quantile(hidden, 0.99, axis=0) + 0.1
        if num_count == 2:
            hidden_maxs /= 2
        bins = np.linspace(hidden_mins, hidden_maxs, num=num_count)
        indexes = np.zeros((hidden.shape[0], hidden.shape[1]))
        for i in range(hidden.shape[1]):
            indexes[:, i] = np.digitize(hidden[:, i], bins[:, i])
        return indexes.tolist()

    def get_pdf(self, if_comp=True):
        x_d = self.discrete_binary(self.X)
        y_d = self.Y.tolist()
        z1_d = self.discrete_hidden(self.Z1, 2)
        z2_d = self.discrete_hidden(self.Z2, 2)
        if if_comp:
            z3_d = self.discrete_binary(self.Z3)
        else:
            z3_d = self.discrete_hidden(self.Z3, 2)
        self.pdf_x = Counter()
        self.pdf_y = Counter()
        self.pdf_z1 = Counter()
        self.pdf_z2 = Counter()
        self.pdf_z3 = Counter()
        self.pdf_xz1 = Counter()
        self.pdf_yz1 = Counter()
        self.pdf_xz2 = Counter()
        self.pdf_yz2 = Counter()
        self.pdf_xz3 = Counter()
        self.pdf_yz3 = Counter()
        for i in range(self.n_samples):
            self.pdf_x[totuple(x_d[i])] += 1 / self.n_samples
            self.pdf_y[totuple(y_d[i])] += 1 / self.n_samples
            self.pdf_z1[totuple(z1_d[i])] += 1 / self.n_samples
            self.pdf_z2[totuple(z2_d[i])] += 1 / self.n_samples
            self.pdf_z3[totuple(z3_d[i])] += 1 / self.n_samples
            self.pdf_xz1[(totuple(x_d[i]), totuple(z1_d[i]))] += 1 / self.n_samples
            self.pdf_yz1[(totuple(y_d[i]), totuple(z1_d[i]))] += 1 / self.n_samples
            self.pdf_xz2[(totuple(x_d[i]), totuple(z2_d[i]))] += 1 / self.n_samples
            self.pdf_yz2[(totuple(y_d[i]), totuple(z2_d[i]))] += 1 / self.n_samples
            self.pdf_xz3[(totuple(x_d[i]), totuple(z3_d[i]))] += 1 / self.n_samples
            self.pdf_yz3[(totuple(y_d[i]), totuple(z3_d[i]))] += 1 / self.n_samples

    def get_mi_with_z(self, pdf_xz, pdf_yz, pdf_z):
        mi_xz = 0
        mi_yz = 0
        for xz in pdf_xz:
            mi_xz += pdf_xz[xz] * np.log(pdf_xz[xz] / (self.pdf_x[xz[0]] * pdf_z[xz[1]]))
        for yz in pdf_yz:
            mi_yz += pdf_yz[yz] * np.log(pdf_yz[yz] / (self.pdf_y[yz[0]] * pdf_z[yz[1]]))
        return mi_xz, mi_yz

    def get_mi(self):
        self.mi_xz1, self.mi_yz1 = self.get_mi_with_z(self.pdf_xz1, self.pdf_yz1, self.pdf_z1)
        self.mi_xz2, self.mi_yz2 = self.get_mi_with_z(self.pdf_xz2, self.pdf_yz2, self.pdf_z2)
        self.mi_xz3, self.mi_yz3 = self.get_mi_with_z(self.pdf_xz3, self.pdf_yz3, self.pdf_z3)

    def get_h(self, pdf):
        h = 0
        for item in pdf:
            h += - pdf[item] * np.log(pdf[item])
        return h

    def save_inf_dict(self, epoch, filename):
        d = {}
        d['mi_xz1'] = self.mi_xz1
        d['mi_yz1'] = self.mi_yz1
        d['mi_xz2'] = self.mi_xz2
        d['mi_yz2'] = self.mi_yz2
        d['mi_xz3'] = self.mi_xz3
        d['mi_yz3'] = self.mi_yz3
        d['h_x'] = self.get_h(self.pdf_x)
        d['h_y'] = self.get_h(self.pdf_y)
        d['h_z1'] = self.get_h(self.pdf_z1)
        d['h_z2'] = self.get_h(self.pdf_z2)
        d['h_z3'] = self.get_h(self.pdf_z3)

        a_file = open(f'../data/{filename}/{epoch}.json', "w")
        json.dump(d, a_file)
        a_file.close()