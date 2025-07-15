#!/usr/bin/env python3
"""
Test search for Anna Dubois specifically in Persons field
"""

import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

def test_anna_dubois_in_persons():
    """Test searching for Anna Dubois in the Persons field structure."""
    print("🔍 Testing Anna Dubois search in Persons field...")
    
    load_dotenv()
    
    try:
        es = Elasticsearch(
            hosts=[os.getenv('ES_HOST')],
            http_auth=(os.getenv('ES_USER'), os.getenv('ES_PASS')),
            verify_certs=False
        )
        
        # Search for Anna Dubois in the Persons field
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        {"match": {"Persons.PersonData.FirstName": "Anna"}},
                        {"match": {"Persons.PersonData.LastName": "Dubois"}},
                        {"match": {"Persons.PersonData.DisplayName": "Anna Dubois"}}
                    ]
                }
            },
            "size": 10
        }
        
        response = es.search(index="research-publications-static", body=search_body)
        hits = response['hits']['hits']
        
        print(f"✅ Found {len(hits)} publications by Anna Dubois")
        
        # Show sample results
        if hits:
            print("\n📋 Sample publications:")
            for i, hit in enumerate(hits[:3], 1):
                source = hit['_source']
                title = source.get('Title', 'Unknown')[:80]
                year = source.get('Year', 'Unknown')
                
                # Extract Anna Dubois from Persons
                persons = source.get('Persons', [])
                anna_person = None
                for person in persons:
                    person_data = person.get('PersonData', {})
                    display_name = person_data.get('DisplayName', '')
                    if 'Anna' in display_name and 'Dubois' in display_name:
                        anna_person = person_data
                        break
                
                if anna_person:
                    print(f"   {i}. {anna_person.get('DisplayName', 'Unknown')} ({year})")
                    print(f"      Title: {title}...")
                    print()
        
        return True, hits
        
    except Exception as e:
        print(f"❌ Search failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, []

if __name__ == "__main__":
    success, results = test_anna_dubois_in_persons()
    if success:
        print(f"🎉 Anna Dubois found! {len(results)} publications")
    else:
        print("❌ Anna Dubois search failed")