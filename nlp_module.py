from transformers import pipeline

 
generator = pipeline("text-generation", model="gpt2")

# Define basic mushroom information
mushroom_info = {
    "Agaricus bisporus": "Agaricus bisporus, also known as the common button mushroom, is one of the most widely consumed mushrooms in the world. It is edible and typically found in fields and grassy areas.",
    "Agrocybe aegerita": "Agrocybe aegerita, commonly known as the poplar mushroom, is an edible species. It grows on wood, especially poplar trees, and has a brown cap and white gills.",
    "Agaricus blazei Murill": "Agaricus blazei Murill, also known as the almond mushroom, is an edible mushroom with medicinal properties. It is cultivated in many parts of the world.",
    "Armillaria mellea": "Armillaria mellea, commonly known as the honey fungus, is a pathogenic fungus that can infect a wide variety of trees and shrubs. It is considered edible but should be thoroughly cooked. This mushroom typically grows on wood.",
    "Auricularia auricula": "Auricularia auricula, also known as the wood ear or black fungus, is an edible mushroom popular in Asian cuisine. It grows on wood, particularly on elder trees.",
    "Auricularia polytricha": "Auricularia polytricha, commonly known as cloud ear or tree ear, is an edible fungus widely used in Chinese cooking. It grows on dead and decaying wood.",
    "Boletus": "Boletus is a genus of mushrooms that includes many species, some of which are highly prized for their edibility. Boletus edulis, also known as the king bolete, is one of the most sought-after wild mushrooms. These mushrooms typically grow in forests."
}

def generate_description(class_name):
    if class_name in mushroom_info:
        base_info = mushroom_info[class_name]
        # Generate a more detailed description using the base information
        detailed_info = generator(base_info, max_length=200, num_return_sequences=1)[0]["generated_text"]
        return detailed_info
    else:
        return "Information about this mushroom class is not available."
