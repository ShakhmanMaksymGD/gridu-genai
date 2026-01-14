"""
Test Script for Synthetic Data Generation Platform

This script tests the core functionality of the platform.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import pandas as pd
from src.utils.ddl_parser import parse_ddl_file, parse_ddl_content
from src.data_generation.synthetic_data_generator import SyntheticDataGenerator
from src.database.postgres_handler import DatabaseHandler
from config.settings import settings

def test_ddl_parser():
    """Test the DDL parser with sample schemas."""
    print("🔍 Testing DDL Parser...")
    
    # Test with company schema
    try:
        schema = parse_ddl_file("samplers/company_employee_schema.ddl")
        print(f"✅ Parsed {len(schema.tables)} tables from company schema")
        
        for table_name, table in schema.tables.items():
            print(f"  - {table_name}: {len(table.columns)} columns, {len(table.constraints)} constraints")
        
        # Test dependency resolution
        creation_order = schema.get_creation_order()
        print(f"✅ Table creation order: {creation_order}")
        
    except Exception as e:
        print(f"❌ DDL Parser test failed: {e}")
        return False
    
    return True

def test_database_handler():
    """Test the database handler functionality."""
    print("🗄️ Testing Database Handler...")
    
    try:
        # Create test schema
        test_ddl = """
        CREATE TABLE test_companies (
            id INT PRIMARY KEY AUTO_INCREMENT,
            name VARCHAR(255) NOT NULL,
            industry VARCHAR(100)
        );
        """
        
        schema = parse_ddl_content(test_ddl)
        db = DatabaseHandler()
        
        # Test connection
        if not db.test_connection():
            print("❌ Database connection failed")
            return False
        
        print("✅ Database connection successful")
        
        # Test schema creation
        success = db.create_schema_tables(schema)
        if success:
            print("✅ Schema creation successful")
        else:
            print("❌ Schema creation failed")
            return False
        
        # Test data insertion
        test_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Company A', 'Company B', 'Company C'],
            'industry': ['Tech', 'Finance', 'Healthcare']
        })
        
        success = db.insert_dataframe('test_companies', test_df)
        if success:
            print("✅ Data insertion successful")
        else:
            print("❌ Data insertion failed")
            return False
        
        # Test data retrieval
        retrieved_df = db.get_table_data('test_companies')
        if len(retrieved_df) == 3:
            print("✅ Data retrieval successful")
        else:
            print(f"❌ Data retrieval failed: expected 3 rows, got {len(retrieved_df)}")
            return False
        
        # Cleanup
        db.drop_all_tables()
        print("✅ Cleanup successful")
        
    except Exception as e:
        print(f"❌ Database Handler test failed: {e}")
        return False
    
    return True

async def test_data_generator():
    """Test the synthetic data generator."""
    print("🤖 Testing Data Generator...")
    
    try:
        # Check if API key is configured
        if not settings.gemini_api_key:
            print("⚠️ GEMINI_API_KEY not configured - skipping data generation test")
            return True
        
        # Create simple test schema
        test_ddl = """
        CREATE TABLE test_users (
            id INT PRIMARY KEY AUTO_INCREMENT,
            name VARCHAR(255) NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            age INT
        );
        """
        
        schema = parse_ddl_content(test_ddl)
        generator = SyntheticDataGenerator()
        
        # Generate small amount of test data
        generated_data = await generator.generate_schema_data(
            schema,
            num_rows_per_table=5,
            user_instructions="Generate realistic test users"
        )
        
        if 'test_users' in generated_data:
            df = generated_data['test_users']
            print(f"✅ Generated {len(df)} rows for test_users")
            print(f"  Columns: {list(df.columns)}")
            return True
        else:
            print("❌ Data generation failed - no data returned")
            return False
    
    except Exception as e:
        print(f"❌ Data Generator test failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive tests on sample schemas."""
    print("🧪 Running Comprehensive Tests...")
    
    sample_files = [
        "samplers/company_employee_schema.ddl",
        "samplers/library_mgm_schema.ddl",
        "samplers/restrurants_schema.ddl"
    ]
    
    for sample_file in sample_files:
        print(f"\n📋 Testing {sample_file}...")
        
        try:
            schema = parse_ddl_file(sample_file)
            print(f"✅ Parsed {len(schema.tables)} tables")
            
            # Test dependency resolution
            creation_order = schema.get_creation_order()
            print(f"✅ Creation order: {creation_order}")
            
            # Validate constraints
            total_constraints = sum(len(table.constraints) for table in schema.tables.values())
            print(f"✅ Found {total_constraints} constraints")
            
        except Exception as e:
            print(f"❌ Failed to test {sample_file}: {e}")

def main():
    """Main test function."""
    print("🚀 Starting Synthetic Data Platform Tests\n")
    
    tests = [
        ("DDL Parser", test_ddl_parser),
        ("Database Handler", test_database_handler),
    ]
    
    results = []
    
    # Run synchronous tests
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        result = test_func()
        results.append((test_name, result))
    
    # Run asynchronous tests
    print(f"\n{'='*50}")
    async_result = asyncio.run(test_data_generator())
    results.append(("Data Generator", async_result))
    
    # Run comprehensive test
    print(f"\n{'='*50}")
    run_comprehensive_test()
    
    # Print summary
    print(f"\n{'='*50}")
    print("📊 Test Summary:")
    print("="*50)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! The platform is ready to use.")
        print("\n🚀 To start the application:")
        print("   - Using Docker: docker-compose up --build")
        print("   - Manually: streamlit run app.py")
        print("\n💡 Don't forget to:")
        print("   - Set your GEMINI_API_KEY in .env")
        print("   - Configure PostgreSQL settings")
        print("   - Optionally set up Langfuse for observability")
    else:
        print("\n❌ Some tests failed. Please check the configuration.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())