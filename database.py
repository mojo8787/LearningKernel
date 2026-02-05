import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.orm import aliased
import json

# Get the database URL from environment variables
# Fall back to SQLite for local development when PostgreSQL is not configured
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///kreinsynergy.db')

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create base class for models
Base = declarative_base()

# Define models
class Antibiotic(Base):
    __tablename__ = 'antibiotics'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    class_name = Column(String(100))
    mechanism = Column(String(200))
    description = Column(Text)
    
    def __repr__(self):
        return f"<Antibiotic(name='{self.name}', class='{self.class_name}')>"

class DrugPair(Base):
    __tablename__ = 'drug_pairs'
    
    id = Column(Integer, primary_key=True)
    drug_a_id = Column(Integer, ForeignKey('antibiotics.id'))
    drug_b_id = Column(Integer, ForeignKey('antibiotics.id'))
    fici = Column(Float)
    synergy_score = Column(Float)
    classification = Column(String(50))
    mic_reduction_a = Column(Float)
    mic_reduction_b = Column(Float)
    growth_inhibition = Column(Float)
    notes = Column(Text)
    
    # Define relationships
    drug_a = relationship("Antibiotic", foreign_keys=[drug_a_id])
    drug_b = relationship("Antibiotic", foreign_keys=[drug_b_id])
    
    def __repr__(self):
        return f"<DrugPair(drug_a='{self.drug_a.name if self.drug_a else None}', drug_b='{self.drug_b.name if self.drug_b else None}')>"

class AnalysisResult(Base):
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    kernel_type = Column(String(50))
    kernel_params = Column(Text)  # Stored as JSON
    visualization_data = Column(Text)  # Stored as JSON
    created_at = Column(String(100))  # Store as ISO format
    
    def __repr__(self):
        return f"<AnalysisResult(name='{self.name}', kernel_type='{self.kernel_type}')>"


# Create the database tables
def init_db():
    Base.metadata.create_all(engine)

# Create a session factory
Session = sessionmaker(bind=engine)

# Function to get a new session
def get_session():
    return Session()

# Functions to interact with the database
def add_antibiotic(name, class_name=None, mechanism=None, description=None):
    session = get_session()
    try:
        antibiotic = Antibiotic(
            name=name,
            class_name=class_name,
            mechanism=mechanism,
            description=description
        )
        session.add(antibiotic)
        session.commit()
        return antibiotic.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def add_drug_pair(drug_a_id, drug_b_id, fici=None, synergy_score=None, classification=None, 
                 mic_reduction_a=None, mic_reduction_b=None, growth_inhibition=None, notes=None):
    session = get_session()
    try:
        drug_pair = DrugPair(
            drug_a_id=drug_a_id,
            drug_b_id=drug_b_id,
            fici=fici,
            synergy_score=synergy_score,
            classification=classification,
            mic_reduction_a=mic_reduction_a,
            mic_reduction_b=mic_reduction_b,
            growth_inhibition=growth_inhibition,
            notes=notes
        )
        session.add(drug_pair)
        session.commit()
        return drug_pair.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def save_analysis_result(name, description, kernel_type, kernel_params, visualization_data, created_at):
    session = get_session()
    try:
        # Convert dictionaries to JSON strings
        kernel_params_json = json.dumps(kernel_params)
        visualization_data_json = json.dumps(visualization_data)
        
        result = AnalysisResult(
            name=name,
            description=description,
            kernel_type=kernel_type,
            kernel_params=kernel_params_json,
            visualization_data=visualization_data_json,
            created_at=created_at
        )
        session.add(result)
        session.commit()
        return result.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_all_antibiotics():
    session = get_session()
    try:
        antibiotics = session.query(Antibiotic).all()
        return antibiotics
    finally:
        session.close()

def get_all_drug_pairs():
    session = get_session()
    try:
        drug_pairs = session.query(DrugPair).all()
        return drug_pairs
    finally:
        session.close()

def get_drug_pairs_as_dataframe():
    session = get_session()
    try:
        # Use aliases to join antibiotics table twice (for drug_a and drug_b)
        AntibioticA = aliased(Antibiotic)
        AntibioticB = aliased(Antibiotic)
        result = session.query(
            DrugPair,
            AntibioticA.name.label('drug_a_name'),
            AntibioticB.name.label('drug_b_name')
        ).join(
            AntibioticA, DrugPair.drug_a_id == AntibioticA.id
        ).join(
            AntibioticB, DrugPair.drug_b_id == AntibioticB.id, isouter=True
        ).all()
        
        # Convert to dataframe
        data = []
        for dp, drug_a_name, drug_b_name in result:
            data.append({
                "Drug_A": drug_a_name,
                "Drug_B": drug_b_name,
                "FICI": dp.fici,
                "Synergy_Score": dp.synergy_score,
                "Classification": dp.classification,
                "MIC_Reduction_A": dp.mic_reduction_a,
                "MIC_Reduction_B": dp.mic_reduction_b,
                "Growth_Inhibition": dp.growth_inhibition
            })
        
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error getting drug pairs as dataframe: {e}")
        # Return empty dataframe as fallback
        return pd.DataFrame(columns=["Drug_A", "Drug_B", "FICI", "Synergy_Score", "Classification", 
                                   "MIC_Reduction_A", "MIC_Reduction_B", "Growth_Inhibition"])
    finally:
        session.close()

def get_all_analysis_results():
    session = get_session()
    try:
        results = session.query(AnalysisResult).all()
        return results
    finally:
        session.close()

def get_analysis_result_by_id(result_id):
    session = get_session()
    try:
        result = session.query(AnalysisResult).filter(AnalysisResult.id == result_id).first()
        if result:
            # Parse JSON strings back to dictionaries
            result.kernel_params = json.loads(result.kernel_params)
            result.visualization_data = json.loads(result.visualization_data)
        return result
    finally:
        session.close()

# Function to seed the database with sample data
def seed_database():
    session = get_session()
    try:
        # Check if we already have data
        if session.query(Antibiotic).count() > 0:
            return  # Database already has data
        
        # Add antibiotic data
        antibiotics = [
            {"name": "Ampicillin", "class_name": "Beta-lactam", "mechanism": "Cell wall synthesis inhibitor"},
            {"name": "Tetracycline", "class_name": "Tetracycline", "mechanism": "Protein synthesis inhibitor"},
            {"name": "Ciprofloxacin", "class_name": "Fluoroquinolone", "mechanism": "DNA gyrase inhibitor"},
            {"name": "Gentamicin", "class_name": "Aminoglycoside", "mechanism": "Protein synthesis inhibitor"},
            {"name": "Trimethoprim", "class_name": "Antifolate", "mechanism": "Folate synthesis inhibitor"},
            {"name": "Erythromycin", "class_name": "Macrolide", "mechanism": "Protein synthesis inhibitor"},
            {"name": "Rifampicin", "class_name": "Rifamycin", "mechanism": "RNA polymerase inhibitor"},
            {"name": "Vancomycin", "class_name": "Glycopeptide", "mechanism": "Cell wall synthesis inhibitor"},
            {"name": "Ceftriaxone", "class_name": "Beta-lactam", "mechanism": "Cell wall synthesis inhibitor"},
            {"name": "Azithromycin", "class_name": "Macrolide", "mechanism": "Protein synthesis inhibitor"}
        ]
        
        # Add antibiotics to database
        for antibiotic in antibiotics:
            session.add(Antibiotic(**antibiotic))
        
        session.commit()
        
        # Get the antibiotic IDs
        antibiotic_ids = {ab.name: ab.id for ab in session.query(Antibiotic).all()}
        
        # Create drug pairs with simulated data
        np.random.seed(42)
        
        # Create all possible pairs
        pairs = []
        names = list(antibiotic_ids.keys())
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                # Convert numpy types to Python native types to avoid database errors
                fici = float(np.random.lognormal(mean=-0.5, sigma=0.7))
                synergy_score = float(1.0 - fici / 5)  # Transform to roughly 0-1 scale
                synergy_score = float(np.clip(synergy_score, 0, 1))
                
                # Determine classification
                if fici > 4:
                    classification = "Antagonistic"
                elif fici >= 0.5:
                    classification = "Additive"
                else:
                    classification = "Synergistic"
                
                # Add other metrics - convert to Python float
                mic_reduction_a = float(np.random.rand() * 10)
                mic_reduction_b = float(np.random.rand() * 10)
                growth_inhibition = float(np.random.rand() * 100)
                
                # Add to database one at a time to avoid bulk insert issues
                drug_pair = DrugPair(
                    drug_a_id=antibiotic_ids[names[i]],
                    drug_b_id=antibiotic_ids[names[j]],
                    fici=fici,
                    synergy_score=synergy_score,
                    classification=classification,
                    mic_reduction_a=mic_reduction_a,
                    mic_reduction_b=mic_reduction_b,
                    growth_inhibition=growth_inhibition
                )
                session.add(drug_pair)
                session.commit()  # Commit each pair individually
        
        print("Database seeded successfully!")
        
    except Exception as e:
        session.rollback()
        print(f"Error seeding database: {e}")
    finally:
        session.close()


# Initialize the database
if __name__ == "__main__":
    init_db()
    seed_database()