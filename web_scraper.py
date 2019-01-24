from bs4 import BeautifulSoup
import requests
import pickle
import string
from utils import *
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# This scrapes the internet to get memes based on the parameters given
# It returns the a list of links to each specific meme (links), links to
# the images of each meme (images), and a number of phrases that work with
# that meme (phrases) (This is based on the num_per_meme variable)
def create_data(num_pages = 1, num_per_meme = 1): # returns links, images, phrases

	# link hypervariables
	page_link = "https://memegenerator.net/memes/popular/alltime/page/"
	page_link_to_meme = "https://memegenerator.net"

	# variables/lists
	links = []
	images = []
	phrases = []

	print("Collecting the links")
	# this takes in the first page and provides each meme's link
	for i in range(1,num_pages+1):
		new_page_link = page_link + str(i)
		page_response = requests.get(new_page_link, timeout = 5)
		soup = BeautifulSoup(page_response.content, "html.parser")
		mydivs = soup.find_all("div", {"class": "char-img"}, recursive=True)
		for i in mydivs:
			children_href = i.find("a" , recursive=False, href = True)['href']
			links.append(page_link_to_meme + str(children_href))

	print("Collecting the text")
	# this takes in the meme links and provides the text for the memes
	for i in links:
		texts = []
		for j in range(1, num_per_meme + 1):
			curr_link = i + "/images/popular/alltime/page/" + str(j)
			curr_page = requests.get(curr_link, timeout = 5)
			soup = BeautifulSoup(curr_page.content, "html.parser")
			main_container = soup.find_all("div", {"class": "optimized-instance-container img"})
			for k in main_container:
				temp_texts = k.find_all({"div"}) # check this
				current_text = ""
				for l in temp_texts:
					current_text += (" " + str(l.text))
				texts.append(current_text[1:])
		filtered_texts = [texts[i] for i in range(len(texts)) if i%2]
		phrases.append(filtered_texts)

	print("Collecting the images")
	# this takes in meme links and provides images
	for i in links:
		curr_link = i + "/images/popular/alltime/page/" + str(1)
		curr_page = requests.get(curr_link, timeout = 5)
		soup = BeautifulSoup(curr_page.content, "html.parser")
		image = soup.find("img", {"class":"shadow3"})
		images.append(image["src"])

	return links, images, phrases

# This takes in the list of links to the images and saves the images to a local
# file and returns the number of images that it saved
def save_images(links): # returns the number of images
	starting_text = "images/image_"
	for index, link in enumerate(links):
		curr_link = starting_text + str(index) + ".jpg"
		with open(curr_link, 'wb') as handle:
			response = requests.get(link, stream=True)

			for block in response.iter_content(1024):
				if not block:
					break

				handle.write(block)
	return len(links)

# This reads the images in the local file and puts them all in a numpy array
# The images are black-and-white (one dimensional) and the dimensions of the
# output are (num_memes * num_per_meme, 60, 60, 1)
def include_images(num_memes, num_per_meme): # returns numpy array of images
	image_files = []
	for i in range(num_memes):
		image_url = "images/image_" + str(i) + ".jpg"
		image = np.array(Image.open(image_url).convert("L"))
		image = np.reshape(image, [60,60,1])
		for i in range(num_per_meme):
			image_files.append(image)
	return np.array(image_files)

# This is a simple function to view the images using matplotlib
def view_all_images(num_memes, num_per_meme, images): # makes a numpy image of the images
	for i in range(num_memes):
		for j in range(num_per_meme):
			plt.subplot(num_memes, num_per_meme, i*num_memes + j + 1)
			plt.imshow(np.reshape(images[i*num_memes + j], [60,60]), cmap = "gray")
	plt.show()

def main():
	with open('data_storage/train_basic.pickle', 'rb') as f:
	    links, image_links, phrases = pickle.load(f)

	cleaned = clean_data(phrases)

	vocabulary = vocabulary_creator(cleaned)

	word2idx, idx2word = dictionary_creator(vocabulary)

	with open("data_storage/train_images_basic.pickle", "rb") as f:
		images = pickle.load(f)

	print(np.shape(images))

	#TODO <-- proces the images
	'''
	- Design the model
	- Preproces the input
	- Run the model on the test set
	'''

if __name__ == "__main__":
	main()