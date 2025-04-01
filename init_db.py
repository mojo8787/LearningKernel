from database import init_db, seed_database

if __name__ == "__main__":
    print("Initializing database...")
    init_db()
    print("Seeding database with sample data...")
    seed_database()
    print("Database setup complete!")