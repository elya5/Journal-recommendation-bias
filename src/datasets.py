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

        'Educational Psychology', 'CBE Life Sciences Education',
        'Interactive Learning Environments', 'Education and Information Technologies',
        'Early Education and Development', 'Contemporary Educational Psychology',
        'Computers and Education', 'Journal of Educational Psychology',
        'British Journal of Educational Technology', 'Higher Education',
        'Journal of School Psychology', 'British Journal of Educational Psychology',
        'Educational Technology Research and Development', 'Teaching and Teacher Education',
        'Learning and Individual Differences',
        'Measurement: Journal of the International Measurement Confederation',
        'European Journal of Psychology of Education', 'Reading and Writing',
        'Social Psychology of Education', 'Nurse Education Today',

        'Soil Biology and Biochemistry', 'Geoderma', 'Soil and Tillage Research',
        'Biology and Fertility of Soils', 'Urban Forestry and Urban Greening',
        'Agricultural Water Management', 'Molecular Plant Pathology',
        'Field Crops Research', 'Environmental Technology and Innovation',
        'Applied Soil Ecology', 'Solid Earth',
        'Soil Dynamics and Earthquake Engineering', 'European Journal of Agronomy',
        'European Journal of Soil Science', 'Land Degradation and Development',
        'Plant and Soil', 'European Journal of Soil Biology',
        'Soil Use and Management', 'Biosystems Engineering', 'Applied Clay Science',
    ])].reset_index(drop=True)
    return df
