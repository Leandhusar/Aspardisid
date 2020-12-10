import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

base_path = './drive/MyDrive/Aspardisid'
longitud, altura = 256, 256
model = base_path + '/Model/model.h5'
weights = base_path + '/Model/weights.h5'

cnn = load_model(model)
cnn.load_weights(weights)

def diseaseRecognize(image):
    x = load_img(image, target_size=(longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = cnn.predict(x)
    result = array[0]
    answer = np.argmax(result)
    return answer

def tratamientoCercospora(afeccion, humedad, altura, clima):
    rutas = [[
      #Muy humedo
      [
          [[0], [4]],    #Tercio inferior
          [[1], [5]],    #Tercio medio
          [[2], [6]]     #Tercio superior
          ],
      #Humedo
      [
          [[0], [4]],    #Tercio inferior
          [[1], [5]],    #Tercio medio
          [[2], [6]]     #Tercio superior
          ],
      #Poco humedo
      [
          [[0], [3]],    #Tercio inferior
          [[1], [5]],    #Tercio medio
          [[2], [6]]     #Tercio superior
          ],
      #Sin humedad percibida
      [
          [[0], [3]],    #Tercio inferior
          [[1], [5]],    #Tercio medio
          [[2], [6]]     #Tercio superior
          ]
      ]]
    tratamientos = {
      0: "Ir a los focos y aplicar producto sistémico para frenar la esporulación (difeconazol).\nEn las partes del cultivo donde no hay pústulas se aplica fungicida protectante.\nRevisar las acumulaciones de humedad y reducirlas.\nRevisar la cantidad de maleza y mantener calles del cultivo limpias de esta.",
      1: "Cambiar a fungicida nativo en lugar de sistémico.\nA el cultivo sin afección se le debe proteger con fungicida sistémico y protectante.\nUtilizar alimentación calcio/boro después de 12 días de uso de los fungicidas mencionados.",
      2: "Cortar y renovar follaje en plantas débiles.\nAmontonar el residuo, tratar con fungicida y realizar dosificación de fungicida protectante sobre el cultivo restante a cosechar.",
      3: "Usar fungicidas en los focos donde se ubican las pústulas del hongo.\nAplicar fungicidas sistémicos de categoría toxicológica baja (línea amarilla o línea azul).",
      4: "Usar fungicidas protectantes, es decir, basados en cobre, zinc, manganeso.\nRevisar cómo aportar en apartados foliares más calcio y boro y el fungicida protectante.\nPuede ser Yodo agrícola, Cardo Bordelés, Manzate.\nReducir la humedad en la tierra drenando el área donde se hospeda el espárrago.",
      5: "Aplicar solo sistémico a el cultivo sin afección.\nA nivel foliar, cambiar la dosificación de potasio y azufre.\nCombinar boro y calcio al momento de aplicar la segunda dosis o en el cambio de fungicida",
      6: "Realizar cosecha temprana, recolectar los tallos con pústulas y fumigarlos con fungicidas sistémicos para evitar esporulación sobre el cultivo sano o nuevo."
    }
    return tratamientos[rutas[afeccion][humedad][altura][clima][0]]

def mostrarOpcionesCercospora():
    print('Clima\n0. Frío\n1. Cálido\n')
    print('Altura de la afección\n0. Tercio inferior\n1. Tercio medio\n2. Tercio superior\n')
    print('Humedad del suelo\n0. Muy húmedo\n1. Húmedo\n2. Poco húmedo\n3. Sin humedad percibida')

def estado(image):
    cercospora = 0
    sana = 1
    afeccion = diseaseRecognize(image)

    if afeccion == sana:
      return [1]
    elif afeccion == cercospora:
      mostrarOpcionesCercospora()
      clima = int(input('Ingrese el clima: '))
      altura = int(input('Ingrese el tercio más alto afectado por la cercospora: '))
      humedad = int(input('Ingrese una descripción de la humedad del suelo: '))
      return tratamientoCercospora(afeccion, humedad, altura, clima)

print("\n", estado(base_path + '/80113789.jpg'))