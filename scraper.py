from bs4 import BeautifulSoup
import requests
import numpy as np
import urllib.request
from sklearn import decomposition
from sklearn.manifold import TSNE
from scipy import misc
# from ggplot import *
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
import json
import pandas as pd

count = 0

# url = "https://www.bookdepository.com/Da-Vinci-Code-Dan-Brown/9781400079179?ref=grid-view&qid=1528371116854&sr=1-2"
headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'}


def library_helper(url, depth, library, info):
    if depth == 3:
        return
    relatives_names = []
    relatives_urls = []
    bookdepo = requests.get(url, headers=headers)
    book_url = bookdepo.url
    bookdepository = BeautifulSoup(bookdepo.text, 'html.parser')
    book_name = bookdepository.find('h1').get_text().strip().replace('&', 'and').replace('#', '').replace('*','').replace('?', '').replace(':', '-').replace('/', ' ').replace('\\', '').replace('"', '').lower().capitalize()
    book_image_url = bookdepository.find('div', 'item-block').find('div', 'item-img').find('img').attrs['src']
    # urllib.request.urlretrieve(book_image_url, book_name + ".jpg")
    for item in bookdepository.find_all('div', 'block-wrap isbn-recommendation-carousel'):
        for book in item.find_all('div', 'book-item'):
            for title in book.find_all('div', 'item-info'):
                link = title.find('a').attrs['href']
                relative = title.find_all('a', href=True)[0].get_text().strip().lower().replace('&', 'and').replace('#','').replace('*', '').replace('?', '').replace(':', '-').replace('/', ' ').replace('\\', '').replace('"','').lower().capitalize()
                if relative not in relatives_names:
                    relatives_names.append(relative)
                    relatives_urls.append("https://www.bookdepository.com" + link)
    library.append(book_name)
    info.append((relatives_names, relatives_urls, book_image_url, book_url))
    for j in range(len(relatives_names)):
        if relatives_names[j] not in library:
            library_helper(relatives_urls[j], depth + 1, library, info)


#
library = []
info = []
library_helper(
    "https://www.bookdepository.com/Alice-Wonderland-Lewis-Carroll/9780486275437?ref=grid-view&qid=1528993414765&sr=1-1",
    0, library, info)

books = []

for i in range(len(library)):
    for book in info[i][0]:
        books.append(book)

books = np.unique(books).tolist()

print(library)
print(len(library))

books_matrix = np.zeros((len(library), len(books)))
books_images = []

for i in range(len(library)):
    image = misc.imread(library[i] + ".jpg", mode='RGB')
    # image = misc.imresize(image, 20)
    books_images.append(image)
    for j in range(len(books)):
        if books[j] in info[i][0]:
            books_matrix[i][j] = 1

booksUrls = []
for i in range(len(library)):
    booksUrls.append(info[i][3])



print(books_matrix.shape)

pca = decomposition.PCA(n_components=50)
pca_results = pca.fit_transform(books_matrix)
print(pca_results)
print(pca_results.shape)
print("==================================")
tsne = TSNE(n_components=2)
axis = pca.components_
tsne_results = tsne.fit_transform(pca_results)
print(tsne_results)
print(tsne_results.shape)

THRESHOLD = 0.7
step_count = 0


def step(points):
    global step_count
    print(step_count)
    step_count += 1
    hasOverlap = False
    m = len(points)
    s = np.zeros((m, 2))
    for i in range(m):
        for j in range(m):
            if i == j: continue
            diff = points[i] - points[j]
            dist = np.sqrt(np.sum(diff ** 2))
            if dist < THRESHOLD:
                hasOverlap = True
                a = np.arctan2(diff[1], diff[0])
                d = 0.2
                s[i] += [d * np.cos(a), d * np.sin(a)]

    # make corrections
    for i in range(m): points[i] += s[i]
    return points, hasOverlap


# points = np.array([[-6.878925, -1.2239769],
#                    [0.96864337, -3.401062],
#                    [-0.5853011, -2.2302558],
#                    [1.9461535, -4.3291364],
#                    [-0.8856069, -1.3244886],
#                    [1.1437066, 9.695767],
#                    [0.81144273, -5.190832],
#                    [-3.3915362, -0.07353584],
#                    [1.1357554, 2.262969],
#                    [-1.4655502, 2.2618244],
#                    [4.1799207, -9.638605],
#                    [2.0339394, -4.223022],
#                    [1.9979706, -6.35537],
#                    [2.2083335, -3.6294053],
#                    [1.6792946, -8.39837],
#                    [-2.2543669, 0.055328],
#                    [2.1636653, -4.938563],
#                    [-11.24849, -1.5325314],
#                    [-12.080431, 6.814822],
#                    [-12.144171, 6.169437],
#                    [-12.527613, 6.8137994],
#                    [-11.256151, 6.471045],
#                    [-12.475608, 5.5300407],
#                    [-8.332321, 7.0516434],
#                    [-13.082112, 5.4567037],
#                    [-2.294086, 12.420704],
#                    [-11.073771, 5.5737486],
#                    [-13.289816, 4.9929],
#                    [-9.224388, 6.489647],
#                    [-12.51664, 4.3079915],
#                    [-3.3010564, 13.787987],
#                    [-9.664826, 4.8614397],
#                    [-8.508721, 8.202181],
#                    [-10.400841, 6.2761807],
#                    [-5.366192, 9.960007],
#                    [-10.37197, 7.16504],
#                    [-9.0313425, 7.6812363],
#                    [9.661136, -3.350919],
#                    [-3.8535614, 13.903809],
#                    [14.808748, -5.788743],
#                    [16.857922, -5.3468986],
#                    [17.004906, -4.3912325],
#                    [13.439632, -6.015508],
#                    [13.674747, -4.5672345],
#                    [14.268641, -5.0026484],
#                    [12.960095, -3.7710545],
#                    [16.002426, -5.811412],
#                    [14.92009, -6.8107853],
#                    [12.404788, -6.026451],
#                    [13.918156, -5.775468],
#                    [12.466063, -5.572963],
#                    [13.493186, -3.6695166],
#                    [16.289072, -4.7020626],
#                    [17.483294, -5.6485424],
#                    [12.8218155, -2.4570243],
#                    [17.897238, -3.3185062],
#                    [18.612352, -4.394519],
#                    [17.784395, -4.9814277],
#                    [13.069091, -2.4470139],
#                    [15.377689, -6.506016],
#                    [-9.134912, 5.4720364],
#                    [-9.809481, 5.218323],
#                    [-5.5094447, 7.744295],
#                    [-11.849756, 3.8350835],
#                    [-3.9217381, 12.439691],
#                    [-5.352525, 6.9355483],
#                    [-7.115067, 8.350651],
#                    [-5.5635066, 0.4543273],
#                    [-5.7963743, 1.5895395],
#                    [-11.324837, -2.5751975],
#                    [-10.304269, 3.4241867],
#                    [-5.8944345, 1.7628967],
#                    [-3.702504, 12.877407],
#                    [-2.294086, 12.420704],
#                    [-4.9235306, 13.701688],
#                    [-4.60153, 14.165448],
#                    [-4.921565, 13.09947],
#                    [-0.26204136, 8.488956],
#                    [-5.376547, 11.956211],
#                    [-3.3531094, 11.942513],
#                    [-4.4211335, 12.414577],
#                    [-3.8985984, 11.507797],
#                    [-5.7972927, 13.590822],
#                    [-6.0702024, 13.057843],
#                    [-1.6332465, 9.093673],
#                    [-4.3712163, 9.024624],
#                    [-12.0998745, -0.7867066],
#                    [-12.596646, -1.1826407],
#                    [-12.511942, -2.9450097],
#                    [-11.742131, -1.6057826],
#                    [-11.892172, -2.7129114],
#                    [-12.1040745, -2.1587677],
#                    [-12.914262, -2.187657],
#                    [-12.928066, -2.685816],
#                    [-8.483208, 0.08651005],
#                    [-8.532612, 8.912987],
#                    [-7.7177925, 8.55473],
#                    [-10.094189, 9.062012],
#                    [-3.733468, 10.582008],
#                    [-8.484547, 0.12820512],
#                    [-0.08382116, 6.3382583],
#                    [0.75488967, 7.3107233],
#                    [0.08746696, 5.647055],
#                    [1.2296791, 6.427555],
#                    [-0.7550175, 8.635558],
#                    [-2.3358135, 3.7368267],
#                    [-3.332531, 3.6757128],
#                    [3.5111735, 6.666203],
#                    [-2.404309, 4.1312],
#                    [-3.2153842, 3.7843056],
#                    [1.1966637, 5.0932193],
#                    [3.0357847, 6.6002135],
#                    [3.3656578, 7.029187],
#                    [3.6643822, 6.822487],
#                    [-2.926834, 8.825957],
#                    [-1.1800576, 9.0051155],
#                    [-2.2613235, 4.8877454],
#                    [-0.7338174, 0.23051041],
#                    [-0.6034596, -0.8559498],
#                    [0.14580198, -3.7420843],
#                    [-0.11586293, 0.43277648],
#                    [0.8339211, -1.3257388],
#                    [-1.3305645, -5.146025],
#                    [4.675901, 1.677382],
#                    [-1.5835122, 0.12363016],
#                    [0.6600857, 0.8451778],
#                    [4.976406, 1.1813705],
#                    [4.2903385, 1.556344],
#                    [5.02076, 0.81617236],
#                    [2.8616736, -0.20099778],
#                    [-2.2217572, -1.8961616],
#                    [-0.68125117, -1.1478636],
#                    [3.0475695, 0.7161513],
#                    [2.7928762, -0.22481595],
#                    [9.512399, -7.402817],
#                    [16.013483, -7.2720947],
#                    [4.6727014, -3.3410077],
#                    [12.898128, -4.765818],
#                    [8.593617, -4.2866707],
#                    [14.727829, -3.4604876],
#                    [15.025594, -5.23395],
#                    [8.864294, -7.42602],
#                    [7.2236853, -5.0163813],
#                    [17.897058, -3.3195317],
#                    [18.612946, -4.3953934],
#                    [-2.1497247, -3.9129019],
#                    [12.026872, -4.249673],
#                    [4.4030704, -4.727336],
#                    [4.5930185, -5.8042893],
#                    [-9.961583, -9.873641],
#                    [4.8749614, -6.122225],
#                    [-9.960955, -9.872597],
#                    [-3.5963454, -2.6334143],
#                    [5.5153646, -5.3414435],
#                    [6.9188933, -5.280015],
#                    [5.4018435, -7.0274734],
#                    [2.2648187, -7.722992],
#                    [2.681206, -7.142818],
#                    [1.886892, -8.344956],
#                    [5.4761024, -8.064559],
#                    [5.6039147, -5.9110184],
#                    [4.284102, -6.3223124],
#                    [-5.1537867, -5.1552544],
#                    [-4.653823, -7.612317],
#                    [-5.7166734, -3.8394632],
#                    [-5.8472285, -4.1114078],
#                    [-5.276564, -3.4707816],
#                    [-2.9516428, -5.7066674],
#                    [-7.4249, -4.8520794],
#                    [-3.1058657, -6.5726104],
#                    [-5.3544683, -3.3332782],
#                    [-7.1495547, -4.6549907],
#                    [-7.530881, -4.91762],
#                    [-3.076727, -6.7666035],
#                    [-2.515335, -6.802573],
#                    [-2.2874591, -8.414336],
#                    [-5.6397643, -8.623632],
#                    [-3.0418112, -5.384063],
#                    [-2.248184, -8.495018],
#                    [-5.276904, -8.4221525],
#                    [-5.5600753, -8.411096],
#                    [-5.0184402, -8.771162]])

hasOverlap = True
loops = 0
while hasOverlap and loops <100:
    points, hasOverlap = step(points)
    loops += 1

# print(tsne_results.tolist())

print(json.dumps(points.T[0].tolist()))
print(json.dumps(points.T[1].tolist()))
# print(booksUrls)

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.set_title("Similar Books", fontsize=20)
# for i, point in enumerate(tsne_results):
#     x, y = point[0], point[1]
#
#     im = OffsetImage(books_images[i], zoom=0.15, cmap='gray')
#     ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
#     ax.scatter(x, y)
#     ax.add_artist(ab)
# ax.update_datalim(np.column_stack([x, y]))
# ax.autoscale()
#
# plt.show()

# df = pd.DataFrame(books_matrix, index=library)
# # df.columns(books)
# df.to_csv("books.csv")
