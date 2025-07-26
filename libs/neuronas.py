import numpy as np


class NeuronaPerceptron:
    def __init__(self, entradas, salidas):
        self.entradas = entradas
        self.salidas = salidas
        self.pesos_Ws = None
        self.bias = None

    def predecir_feedforward(self, muestra_datos):
        if self.__is_one_dimension(muestra_datos):
            return self.__activacion(self.__sumatoria(muestra_datos))

        return self.__activacion_lote(self.__sumatoria_lote(muestra_datos))

    def __is_one_dimension(self, muestra_datos):
        return isinstance(muestra_datos, np.ndarray) and muestra_datos.ndim == 1

    def __iniciar_pesos_y_bias(self):
        self.pesos_Ws = np.random.rand(self.entradas)

        # self.pesos_Ws = []
        # for _ in range(self.entradas):
        #     valor_peso_actual = np.random.rand()
        #     self.pesos_Ws.append(valor_peso_actual)

        self.bias = np.random.rand()

    def __sumatoria(self, muestra_datos):
        return np.dot(self.pesos_Ws, muestra_datos) + self.bias

        # z_sumatoria_muestra_datos = 0
        # for i in range(len(muestra_datos)):
        #     entrada_Xi = muestra_datos[i]
        #     peso_Wi = self.pesos_Ws[i]
        #     z_sumatoria_muestra_datos = z_sumatoria_muestra_datos + (
        #         peso_Wi * entrada_Xi
        #     )
        # z_sumatoria_muestra_datos = z_sumatoria_muestra_datos + self.bias
        # return z_sumatoria_muestra_datos

    def __sumatoria_lote(self, muestra_datos):
        return [self.__sumatoria(subarray) for subarray in muestra_datos]

        # mis_Zs = []
        # # Hay que iterar sobre todas las filas, y fila por fila.
        # for i in range(len(muestra_datos[:, 0])):
        #     # Posteriormente, ubicado en cada fila, mandar a llamar a "__sumatoria".
        #     muestra_datos_i = muestra_datos[i, :]
        #     z_sumatoria_muestra_datos = self.__sumatoria(muestra_datos_i)
        #     mis_Zs.append(z_sumatoria_muestra_datos)
        # return mis_Zs

    def __activacion(self, Z):
        if Z >= 0:
            return 1

        return 0

    def __activacion_lote(self, Zs):
        return [self.__activacion(z) for z in Zs]

        # mis_As = []
        # for z in Zs:
        #     mis_As.append(self.__activacion(z))
        # return mis_As
