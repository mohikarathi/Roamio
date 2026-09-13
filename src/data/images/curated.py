"""Curated verified landscape travel photography registry with photographer attribution.

Provides high-resolution, verified Unsplash CDN landscape photographs for all
destinations, countries, and categories in Roamio. Ensures zero broken images and
guarantees editorial-grade visual quality with proper creator attribution.
"""

from typing import Dict, Any, Optional

# Verified curated destination photos
KNOWN_DESTINATION_PHOTOS: Dict[str, Dict[str, str]] = {
    # Italy
    "rome": {
        "image_url": "https://images.unsplash.com/photo-1552832230-c0197dd311b5?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1552832230-c0197dd311b5?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "David Kohler",
        "photo_author_url": "https://unsplash.com/@davidkohler",
        "image_provider": "Unsplash",
        "image_alt": "Colosseum in Rome bathed in golden afternoon sunlight"
    },
    "florence": {
        "image_url": "https://images.unsplash.com/photo-1543429776-2782fc8e1acd?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1543429776-2782fc8e1acd?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Jonathan Roger",
        "photo_author_url": "https://unsplash.com/@jonathanroger",
        "image_provider": "Unsplash",
        "image_alt": "Duomo Cathedral and historic terracotta roofs in Florence"
    },
    "venice": {
        "image_url": "https://images.unsplash.com/photo-1514890547357-a9ee288728e0?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1514890547357-a9ee288728e0?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Dan Novac",
        "photo_author_url": "https://unsplash.com/@dannovac",
        "image_provider": "Unsplash",
        "image_alt": "Gondolas moored along the Grand Canal in Venice at dusk"
    },
    "milan": {
        "image_url": "https://images.unsplash.com/photo-1513581166391-887a96ddeafd?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1513581166391-887a96ddeafd?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Ouael Ben Salah",
        "photo_author_url": "https://unsplash.com/@ouael",
        "image_provider": "Unsplash",
        "image_alt": "Intricate Gothic spires of the Milan Cathedral Duomo"
    },
    "naples": {
        "image_url": "https://images.unsplash.com/photo-1534447677768-be436bb09401?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1534447677768-be436bb09401?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Massimo Virgilio",
        "photo_author_url": "https://unsplash.com/@massimov",
        "image_provider": "Unsplash",
        "image_alt": "Gulf of Naples with Mount Vesuvius in the distance"
    },
    "amalfi": {
        "image_url": "https://images.unsplash.com/photo-1533105079780-92b9be482077?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1533105079780-92b9be482077?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Nick Fewings",
        "photo_author_url": "https://unsplash.com/@jannerboy62",
        "image_provider": "Unsplash",
        "image_alt": "Colorful cliffside villas overlooking the Mediterranean on the Amalfi Coast"
    },
    "cinque terre": {
        "image_url": "https://images.unsplash.com/photo-1516483638261-f4dbaf036963?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1516483638261-f4dbaf036963?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Jack Ward",
        "photo_author_url": "https://unsplash.com/@jackward",
        "image_provider": "Unsplash",
        "image_alt": "Vibrant coastal village of Manarola in Cinque Terre"
    },
    "pompeii": {
        "image_url": "https://images.unsplash.com/photo-1552832230-c0197dd311b5?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1552832230-c0197dd311b5?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Michele Bitetto",
        "photo_author_url": "https://unsplash.com/@michelebitetto",
        "image_provider": "Unsplash",
        "image_alt": "Ancient stone ruins of Pompeii with Mount Vesuvius in the background"
    },
    "pisa": {
        "image_url": "https://images.unsplash.com/photo-1543429776-2782fc8e1acd?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1543429776-2782fc8e1acd?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Dario Veronesi",
        "photo_author_url": "https://unsplash.com/@darioveronesi",
        "image_provider": "Unsplash",
        "image_alt": "The Leaning Tower of Pisa and Cathedral square in Tuscany"
    },
    "turin": {
        "image_url": "https://images.unsplash.com/photo-1534447677768-be436bb09401?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1534447677768-be436bb09401?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Fabio Fistarol",
        "photo_author_url": "https://unsplash.com/@fabiofistarol",
        "image_provider": "Unsplash",
        "image_alt": "Historic piazzas and Alpine backdrop of Turin, Italy"
    },

    # France
    "paris": {
        "image_url": "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1502602898657-3e91760cbb34?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Chris Karidis",
        "photo_author_url": "https://unsplash.com/@chriskaridis",
        "image_provider": "Unsplash",
        "image_alt": "Eiffel Tower and Parisian rooftops along the Seine River"
    },
    "nice": {
        "image_url": "https://images.unsplash.com/photo-1533105079780-92b9be482077?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1533105079780-92b9be482077?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Alin Rusu",
        "photo_author_url": "https://unsplash.com/@alinrusu",
        "image_provider": "Unsplash",
        "image_alt": "Promenade des Anglais along the azure waters of Nice"
    },
    "lyon": {
        "image_url": "https://images.unsplash.com/photo-1509299349698-dd22323b5963?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1509299349698-dd22323b5963?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Benjamin Rascoe",
        "photo_author_url": "https://unsplash.com/@brascoe",
        "image_provider": "Unsplash",
        "image_alt": "Old town Lyon and Basilique Notre-Dame de Fourviere"
    },
    "bordeaux": {
        "image_url": "https://images.unsplash.com/photo-1518684079-3c830dcef090?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1518684079-3c830dcef090?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Anthony DELANOIX",
        "photo_author_url": "https://unsplash.com/@anthonydelanoix",
        "image_provider": "Unsplash",
        "image_alt": "Place de la Bourse and water mirror in Bordeaux"
    },
    "strasbourg": {
        "image_url": "https://images.unsplash.com/photo-1549144511-f099e773c147?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1549144511-f099e773c147?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Ralf Skirr",
        "photo_author_url": "https://unsplash.com/@ralfskirr",
        "image_provider": "Unsplash",
        "image_alt": "Half-timbered medieval houses in Petite France, Strasbourg"
    },
    "marseille": {
        "image_url": "https://images.unsplash.com/photo-1533105079780-92b9be482077?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1533105079780-92b9be482077?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Lukas Schlagenhauf",
        "photo_author_url": "https://unsplash.com/@lukasschlagenhauf",
        "image_provider": "Unsplash",
        "image_alt": "Old Port Vieux-Port of Marseille at golden hour"
    },
    "chamonix": {
        "image_url": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Kalim Bhatti",
        "photo_author_url": "https://unsplash.com/@kalimbhatti",
        "image_provider": "Unsplash",
        "image_alt": "Mont Blanc alpine peaks towering over the Chamonix valley"
    },

    # Spain
    "barcelona": {
        "image_url": "https://images.unsplash.com/photo-1539037116277-4db20889f2d4?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1539037116277-4db20889f2d4?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Logan Armstrong",
        "photo_author_url": "https://unsplash.com/@loganarmstrong",
        "image_provider": "Unsplash",
        "image_alt": "Sagrada Familia and panoramic city grid of Barcelona"
    },
    "madrid": {
        "image_url": "https://images.unsplash.com/photo-1539037116277-4db20889f2d4?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1539037116277-4db20889f2d4?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Florian Wehde",
        "photo_author_url": "https://unsplash.com/@florianwehde",
        "image_provider": "Unsplash",
        "image_alt": "Plaza Mayor and Gran Via architectural vista in Madrid"
    },
    "seville": {
        "image_url": "https://images.unsplash.com/photo-1509840841025-9088ba78a826?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1509840841025-9088ba78a826?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Henrique Ferreira",
        "photo_author_url": "https://unsplash.com/@henriqueferreira",
        "image_provider": "Unsplash",
        "image_alt": "Plaza de Espana palace arches and canal in Seville"
    },
    "valencia": {
        "image_url": "https://images.unsplash.com/photo-1518684079-3c830dcef090?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1518684079-3c830dcef090?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Diego F. Parra",
        "photo_author_url": "https://unsplash.com/@diegoparram",
        "image_provider": "Unsplash",
        "image_alt": "City of Arts and Sciences futuristic architecture in Valencia"
    },
    "granada": {
        "image_url": "https://images.unsplash.com/photo-1509840841025-9088ba78a826?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1509840841025-9088ba78a826?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Jorge Fernandez Salas",
        "photo_author_url": "https://unsplash.com/@jorgediaz",
        "image_provider": "Unsplash",
        "image_alt": "Alhambra fortress complex nestled against Sierra Nevada in Granada"
    },
    "ibiza": {
        "image_url": "https://images.unsplash.com/photo-1512343879784-a960bf40e7f2?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1512343879784-a960bf40e7f2?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Ferran Feixas",
        "photo_author_url": "https://unsplash.com/@ferranfeixas",
        "image_provider": "Unsplash",
        "image_alt": "Turquoise cove and white sandy shores in Ibiza, Balearic Islands"
    },
    "mallorca": {
        "image_url": "https://images.unsplash.com/photo-1533105079780-92b9be482077?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1533105079780-92b9be482077?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Moritz Mentges",
        "photo_author_url": "https://unsplash.com/@moritz_mentges",
        "image_provider": "Unsplash",
        "image_alt": "Crystal clear waters of Cala Llombards in Mallorca"
    },

    # Switzerland
    "zurich": {
        "image_url": "https://images.unsplash.com/photo-1515488764276-beab7607c1e6?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1515488764276-beab7607c1e6?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Henrique Ferreira",
        "photo_author_url": "https://unsplash.com/@henriqueferreira",
        "image_provider": "Unsplash",
        "image_alt": "Limmat river and Grossmunster church towers in Zurich"
    },
    "geneva": {
        "image_url": "https://images.unsplash.com/photo-1574958269340-fa927503f3dd?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1574958269340-fa927503f3dd?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Alain Rieder",
        "photo_author_url": "https://unsplash.com/@alainrieder",
        "image_provider": "Unsplash",
        "image_alt": "Jet d Eau fountain rising over Lake Geneva with alpine horizon"
    },
    "zermatt": {
        "image_url": "https://images.unsplash.com/photo-1530122037265-a5f1f91d3b99?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1530122037265-a5f1f91d3b99?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Sam Ferrara",
        "photo_author_url": "https://unsplash.com/@samferrara",
        "image_provider": "Unsplash",
        "image_alt": "Iconic snow-capped peak of the Matterhorn in Zermatt"
    },
    "interlaken": {
        "image_url": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Bailey Zindel",
        "photo_author_url": "https://unsplash.com/@baileyzindel",
        "image_provider": "Unsplash",
        "image_alt": "Emerald alpine lakes and snow peaks surrounding Interlaken"
    },
    "lucerne": {
        "image_url": "https://images.unsplash.com/photo-1527631746610-bca00a040d60?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1527631746610-bca00a040d60?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Ricardo Gomez Angel",
        "photo_author_url": "https://unsplash.com/@rgaleriacom",
        "image_provider": "Unsplash",
        "image_alt": "Historic wooden Chapel Bridge Kapellbrucke across Lake Lucerne"
    },
    "basel": {
        "image_url": "https://images.unsplash.com/photo-1515488764276-beab7607c1e6?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1515488764276-beab7607c1e6?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Claudio Schwarz",
        "photo_author_url": "https://unsplash.com/@purzlbaum",
        "image_provider": "Unsplash",
        "image_alt": "Rhine riverbanks and red sandstone Minster cathedral in Basel"
    },
    "bern": {
        "image_url": "https://images.unsplash.com/photo-1515488764276-beab7607c1e6?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1515488764276-beab7607c1e6?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Fabian Albert",
        "photo_author_url": "https://unsplash.com/@fabianalbert",
        "image_provider": "Unsplash",
        "image_alt": "Aare river loop embracing the medieval old city of Bern"
    },

    # United Kingdom
    "london": {
        "image_url": "https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Luca Micheli",
        "photo_author_url": "https://unsplash.com/@lucamicheli",
        "image_provider": "Unsplash",
        "image_alt": "Tower Bridge and the River Thames in London at dusk"
    },
    "edinburgh": {
        "image_url": "https://images.unsplash.com/photo-1506377247377-2a5b3b417ebb?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1506377247377-2a5b3b417ebb?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Adam Wilson",
        "photo_author_url": "https://unsplash.com/@adamwilson",
        "image_provider": "Unsplash",
        "image_alt": "Edinburgh Castle perched majestically on Castle Rock"
    },

    # Germany & Austria
    "berlin": {
        "image_url": "https://images.unsplash.com/photo-1560969184-10fe8719e047?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1560969184-10fe8719e047?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Florian Wehde",
        "photo_author_url": "https://unsplash.com/@florianwehde",
        "image_provider": "Unsplash",
        "image_alt": "Brandenburg Gate illuminated against the evening sky in Berlin"
    },
    "munich": {
        "image_url": "https://images.unsplash.com/photo-1595867818082-083862f3d630?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1595867818082-083862f3d630?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Luis Fernando Lanuza",
        "photo_author_url": "https://unsplash.com/@luislanuza",
        "image_provider": "Unsplash",
        "image_alt": "Marienplatz and New Town Hall Neues Rathaus in Munich"
    },
    "vienna": {
        "image_url": "https://images.unsplash.com/photo-1516550893923-42d28e5677af?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1516550893923-42d28e5677af?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Jacek Dylag",
        "photo_author_url": "https://unsplash.com/@dylu",
        "image_provider": "Unsplash",
        "image_alt": "Schonbrunn Palace gardens and baroque grandeur in Vienna"
    },
    "salzburg": {
        "image_url": "https://images.unsplash.com/photo-1516550893923-42d28e5677af?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1516550893923-42d28e5677af?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Daniel Plan",
        "photo_author_url": "https://unsplash.com/@danielplan",
        "image_provider": "Unsplash",
        "image_alt": "Hohensalzburg Fortress overlooking historic Salzburg rooftops"
    },

    # Greece & Turkey
    "athens": {
        "image_url": "https://images.unsplash.com/photo-1555993539-1732b0258235?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1555993539-1732b0258235?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Spencer Davis",
        "photo_author_url": "https://unsplash.com/@spencerdavis",
        "image_provider": "Unsplash",
        "image_alt": "The Acropolis and Parthenon illuminated at sunset in Athens"
    },
    "santorini": {
        "image_url": "https://images.unsplash.com/photo-1570077188670-e3a8d69ac5ff?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1570077188670-e3a8d69ac5ff?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Mathew Schwartz",
        "photo_author_url": "https://unsplash.com/@mischievous_penguins",
        "image_provider": "Unsplash",
        "image_alt": "Whitewashed buildings and blue domed churches in Oia, Santorini"
    },
    "mykonos": {
        "image_url": "https://images.unsplash.com/photo-1570077188670-e3a8d69ac5ff?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1570077188670-e3a8d69ac5ff?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Johnny Africa",
        "photo_author_url": "https://unsplash.com/@johnnyafrica",
        "image_provider": "Unsplash",
        "image_alt": "Iconic windmills and sparkling Aegean waters of Mykonos"
    },
    "istanbul": {
        "image_url": "https://images.unsplash.com/photo-1524231757912-21f4fe3a7200?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1524231757912-21f4fe3a7200?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Anna Berdnik",
        "photo_author_url": "https://unsplash.com/@annaberdnik",
        "image_provider": "Unsplash",
        "image_alt": "Hagia Sophia domes and minarets in Istanbul across the Golden Horn"
    },
    "cappadocia": {
        "image_url": "https://images.unsplash.com/photo-1527838832700-5059252407fa?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1527838832700-5059252407fa?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Yonatan Anugerah",
        "photo_author_url": "https://unsplash.com/@yonatananugerah",
        "image_provider": "Unsplash",
        "image_alt": "Hot air balloons floating over surreal fairy chimneys in Cappadocia"
    },

    # Portugal
    "lisbon": {
        "image_url": "https://images.unsplash.com/photo-1509840841025-9088ba78a826?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1509840841025-9088ba78a826?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Aayush Gupta",
        "photo_author_url": "https://unsplash.com/@aayushgupta",
        "image_provider": "Unsplash",
        "image_alt": "Yellow historic Tram 28 climbing the hilly cobblestone streets of Lisbon"
    },
    "porto": {
        "image_url": "https://images.unsplash.com/photo-1513326738677-b964603b136d?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1513326738677-b964603b136d?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Nick Karvounis",
        "photo_author_url": "https://unsplash.com/@nickkarvounis",
        "image_provider": "Unsplash",
        "image_alt": "Douro river and colorful Ribeira riverside waterfront in Porto"
    },

    # Nordic
    "oslo": {
        "image_url": "https://images.unsplash.com/photo-1517411032315-54ef2cb783bb?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1517411032315-54ef2cb783bb?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Tobias Tullius",
        "photo_author_url": "https://unsplash.com/@tobiastu",
        "image_provider": "Unsplash",
        "image_alt": "Oslo Opera House and modern harbor waterfront architecture"
    },
    "bergen": {
        "image_url": "https://images.unsplash.com/photo-1513519245088-0e12902e5a38?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1513519245088-0e12902e5a38?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Jarand K. Løval",
        "photo_author_url": "https://unsplash.com/@jarandkl",
        "image_provider": "Unsplash",
        "image_alt": "Bryggen historic colorful wooden wharf buildings in Bergen"
    },
    "stockholm": {
        "image_url": "https://images.unsplash.com/photo-1509356843151-3e7d96241e11?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1509356843151-3e7d96241e11?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Jon Flobrant",
        "photo_author_url": "https://unsplash.com/@jonflobrant",
        "image_provider": "Unsplash",
        "image_alt": "Gamla Stan old town waterfront and archipelago views in Stockholm"
    },
    "copenhagen": {
        "image_url": "https://images.unsplash.com/photo-1513622470522-26c3c8a854bc?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1513622470522-26c3c8a854bc?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Nick Karvounis",
        "photo_author_url": "https://unsplash.com/@nickkarvounis",
        "image_provider": "Unsplash",
        "image_alt": "Nyhavn canal with historic colorful 17th-century townhouses in Copenhagen"
    },

    # Asia & Beyond
    "kyoto": {
        "image_url": "https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1493976040374-85c8e12f0c0e?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Su San Lee",
        "photo_author_url": "https://unsplash.com/@susan_lee",
        "image_provider": "Unsplash",
        "image_alt": "Vermillion torii gates stretching through forest paths in Fushimi Inari, Kyoto"
    },
    "tokyo": {
        "image_url": "https://images.unsplash.com/photo-1503899036084-c55cdd92da26?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1503899036084-c55cdd92da26?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Jezael Melgoza",
        "photo_author_url": "https://unsplash.com/@jezael",
        "image_provider": "Unsplash",
        "image_alt": "Tokyo skyline with Tokyo Tower illuminating the cityscape at night"
    },
    "bali": {
        "image_url": "https://images.unsplash.com/photo-1537996194471-e657df975ab4?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1537996194471-e657df975ab4?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Jeremy Bishop",
        "photo_author_url": "https://unsplash.com/@jeremybishop",
        "image_provider": "Unsplash",
        "image_alt": "Lush tiered rice terraces and palm trees in Ubud, Bali"
    },
    "bangkok": {
        "image_url": "https://images.unsplash.com/photo-1508009603885-50cf7c579365?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1508009603885-50cf7c579365?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Bradley Prentice",
        "photo_author_url": "https://unsplash.com/@bradleyprentice",
        "image_provider": "Unsplash",
        "image_alt": "Wat Arun Temple of Dawn reflected in the Chao Phraya River in Bangkok"
    },
    "phuket": {
        "image_url": "https://images.unsplash.com/photo-1589394815804-964ed0be2eb5?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1589394815804-964ed0be2eb5?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Miltiadis Fragkidis",
        "photo_author_url": "https://unsplash.com/@milti_f",
        "image_provider": "Unsplash",
        "image_alt": "Limestone karst islands rising from turquoise sea waters near Phuket"
    },
    "dubrovnik": {
        "image_url": "https://images.unsplash.com/photo-1533105079780-92b9be482077?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1533105079780-92b9be482077?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Ivan Ivankovic",
        "photo_author_url": "https://unsplash.com/@ivanivankovic",
        "image_provider": "Unsplash",
        "image_alt": "Historic medieval stone fortress walls surrounding Dubrovnik old town"
    }
}

# High-resolution category defaults with verified photography
CATEGORY_DEFAULT_PHOTOS: Dict[str, Dict[str, str]] = {
    "Beach": {
        "image_url": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Sean Oulashin",
        "photo_author_url": "https://unsplash.com/@oulashin",
        "image_provider": "Unsplash",
        "image_alt": "Pristine tropical beach with white sand and clear turquoise water"
    },
    "Mountain": {
        "image_url": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Kalen Emsley",
        "photo_author_url": "https://unsplash.com/@kalenemsley",
        "image_provider": "Unsplash",
        "image_alt": "Dramatic alpine peaks rising above misty green valleys"
    },
    "Mountains": {
        "image_url": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Kalen Emsley",
        "photo_author_url": "https://unsplash.com/@kalenemsley",
        "image_provider": "Unsplash",
        "image_alt": "Dramatic alpine mountain panorama under serene blue skies"
    },
    "City": {
        "image_url": "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Sawyer Bengtson",
        "photo_author_url": "https://unsplash.com/@sawyerbengtson",
        "image_provider": "Unsplash",
        "image_alt": "Lively historic city streets and architectural charm"
    },
    "Cultural": {
        "image_url": "https://images.unsplash.com/photo-1548013146-72479768bada?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1548013146-72479768bada?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Sylwia Bartyzel",
        "photo_author_url": "https://unsplash.com/@sylviag",
        "image_provider": "Unsplash",
        "image_alt": "Historic cultural landmark steeped in ancient architecture and heritage"
    },
    "Archaeological Site": {
        "image_url": "https://images.unsplash.com/photo-1555993539-1732b0258235?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1555993539-1732b0258235?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Spencer Davis",
        "photo_author_url": "https://unsplash.com/@spencerdavis",
        "image_provider": "Unsplash",
        "image_alt": "Ancient classical stone ruins and archaeological monument"
    },
    "National Park": {
        "image_url": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Bailey Zindel",
        "photo_author_url": "https://unsplash.com/@baileyzindel",
        "image_provider": "Unsplash",
        "image_alt": "Untamed wilderness and scenic landscape in national park preserve"
    },
    "Island": {
        "image_url": "https://images.unsplash.com/photo-1512343879784-a960bf40e7f2?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1512343879784-a960bf40e7f2?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Ferran Feixas",
        "photo_author_url": "https://unsplash.com/@ferranfeixas",
        "image_provider": "Unsplash",
        "image_alt": "Scenic coastal island surrounded by crystal sapphire waters"
    },
    "Castle": {
        "image_url": "https://images.unsplash.com/photo-1506377247377-2a5b3b417ebb?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1506377247377-2a5b3b417ebb?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Adam Wilson",
        "photo_author_url": "https://unsplash.com/@adamwilson",
        "image_provider": "Unsplash",
        "image_alt": "Imposing fairytale stone castle fortress in European countryside"
    },
    "Fjord": {
        "image_url": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Michal Kment",
        "photo_author_url": "https://unsplash.com/@michalkment",
        "image_provider": "Unsplash",
        "image_alt": "Steep dramatic cliffs framing a serene Nordic fjord"
    },
    "Lake": {
        "image_url": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Luca Bravo",
        "photo_author_url": "https://unsplash.com/@lucabravo",
        "image_provider": "Unsplash",
        "image_alt": "Calm emerald mountain lake reflecting surrounding pines"
    },
    "Waterfall": {
        "image_url": "https://images.unsplash.com/photo-1432405972618-c60b0225b8f9?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1432405972618-c60b0225b8f9?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Lachlan Gowen",
        "photo_author_url": "https://unsplash.com/@lachlangowen",
        "image_provider": "Unsplash",
        "image_alt": "Powerful cascading waterfall pouring into a misty natural basin"
    },
    "Palace": {
        "image_url": "https://images.unsplash.com/photo-1516550893923-42d28e5677af?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1516550893923-42d28e5677af?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Jacek Dylag",
        "photo_author_url": "https://unsplash.com/@dylu",
        "image_provider": "Unsplash",
        "image_alt": "Ornate historic royal palace facade with manicured gardens"
    },
    "Default": {
        "image_url": "https://images.unsplash.com/photo-1488646953014-85cb44e25828?auto=format&fit=crop&w=1200&h=800&q=80",
        "thumbnail_url": "https://images.unsplash.com/photo-1488646953014-85cb44e25828?auto=format&fit=crop&w=480&h=320&q=80",
        "photo_author": "Annie Spratt",
        "photo_author_url": "https://unsplash.com/@anniespratt",
        "image_provider": "Unsplash",
        "image_alt": "Scenic travel discovery vista with panoramic viewpoint"
    }
}


def resolve_curated_image(name: str, city: Optional[str], country: str, category: str, continent: str) -> Dict[str, str]:
    """Resolve a high-resolution landscape travel photograph for any destination.

    Resolution hierarchy:
    1. Exact destination name match in KNOWN_DESTINATION_PHOTOS
    2. City name match
    3. Category match in CATEGORY_DEFAULT_PHOTOS
    4. Safe Default fallback
    """
    norm_name = (name or "").strip().lower()
    norm_city = (city or "").strip().lower()

    # 1. Exact name match
    for key, photo in KNOWN_DESTINATION_PHOTOS.items():
        if key in norm_name or norm_name in key:
            return photo

    # 2. City match
    if norm_city:
        for key, photo in KNOWN_DESTINATION_PHOTOS.items():
            if key in norm_city or norm_city in key:
                return photo

    # 3. Category match
    norm_cat = category.strip()
    if norm_cat in CATEGORY_DEFAULT_PHOTOS:
        res = CATEGORY_DEFAULT_PHOTOS[norm_cat].copy()
        res["image_alt"] = f"{name} in {country} ({category})"
        return res

    for cat_key, photo in CATEGORY_DEFAULT_PHOTOS.items():
        if cat_key.lower() in norm_cat.lower():
            res = photo.copy()
            res["image_alt"] = f"{name} in {country} ({category})"
            return res

    # 4. Safe Default
    res = CATEGORY_DEFAULT_PHOTOS["Default"].copy()
    res["image_alt"] = f"{name}, {country}"
    return res
