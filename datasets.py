import pandas as pd

RANDOM_STATE = 42

def get_articles(key='infectious_diseases'):
    df = pd.read_feather(f'data/articles_{key}.feather')
    df.dropna(subset=['abstract', 'title'], inplace=True)
    df = pd.concat([
        df[df.type == 'train'],
        df[df.type=='test'].groupby('journal_id').sample(100, random_state=RANDOM_STATE)
    ])
    df = df[df.journal_title.isin([
        'Viruses', 'Clinical Infectious Diseases', 'BMC Infectious Diseases', 
        'Vaccine', 'Pathogens', 'Antibiotics', 'PLOS Neglected Tropical Diseases', 
        'Frontiers in Cellular and Infection Microbiology', 'Vaccines', 
        'The Journal of Infectious Diseases', 
        'Journal of Medical Virology', 'Parasites & Vectors', 
        'Parasitology Research', 'Microbial Pathogenesis',
        'Infection and Drug Resistance', 'Malaria Journal', 
        'Clinical Microbiology and Infection', 'Infection Genetics and Evolution', 
        'The Pediatric Infectious Disease Journal',  'Journal of Antimicrobial Chemotherapy',

        'Materials Science and Engineering A', 'Materials Letters',
        'Advanced Materials', 'Journal of Materials Science', 'Journal of Alloys and Compounds',
        'Materials Science and Engineering C', 'Composites Part B Engineering',
        'Materials & Design', 'Materials Today Communications', 
        'Journal of Materials Engineering and Performance', 
        'Microporous and Mesoporous Materials', 'Computational Materials Science', 
        'Advanced Materials Interfaces', 'Materials Characterization', 
        'Journal of Material Science and Technology',  'Metallurgical and Materials Transactions A',
        'Biomaterials', 'Materials Science in Semiconductor Processing', 
        'Journal of the Mechanical Behavior of Biomedical Materials', 
        'Journal of Materials in Civil Engineering', 

    ])].reset_index(drop=True)
    return df
