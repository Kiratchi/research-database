#!/usr/bin/env python3
"""
Debug search tools to find why Anna Dubois returns 0 results
"""

import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

def debug_database_structure():
    """Debug the actual database structure."""
    print("🔍 Debugging database structure...")
    
    load_dotenv()
    
    try:
        es = Elasticsearch(
            hosts=[os.getenv('ES_HOST')],
            http_auth=(os.getenv('ES_USER'), os.getenv('ES_PASS')),
            verify_certs=False
        )
        
        # Get index mapping
        print("\n📋 Getting index mapping...")
        mapping = es.indices.get_mapping(index="research-publications-static")
        
        # Extract field names (ES 6.8 structure)
        print(f"🔍 Raw mapping structure: {json.dumps(mapping, indent=2)[:500]}...")
        
        # Try different mapping structures
        try:
            # ES 7+ structure
            fields = mapping["research-publications-static"]["mappings"]["properties"].keys()
        except KeyError:
            try:
                # ES 6.8 structure with doc type
                doc_type = list(mapping["research-publications-static"]["mappings"].keys())[0]
                fields = mapping["research-publications-static"]["mappings"][doc_type]["properties"].keys()
            except (KeyError, IndexError):
                # Fallback: extract from sample document
                print("⚠️ Cannot extract fields from mapping, using sample document")
                sample_response = es.search(index="research-publications-static", body={"query": {"match_all": {}}, "size": 1})
                if sample_response['hits']['hits']:
                    fields = sample_response['hits']['hits'][0]['_source'].keys()
                else:
                    fields = []
        
        print(f"✅ Available fields: {list(fields)}")
        
        # Get a sample document with actual data
        print("\n📄 Getting sample documents with actual data...")
        sample_query = {
            "query": {"match_all": {}},
            "size": 10
        }
        
        response = es.search(index="research-publications-static", body=sample_query)
        
        # Look for documents that have actual author data
        for hit in response['hits']['hits']:
            source = hit['_source']
            author_field = source.get('author', 'NOT_FOUND')
            authors_field = source.get('authors', 'NOT_FOUND')
            title = source.get('title', 'NOT_FOUND')
            
            if author_field != 'NOT_FOUND' and author_field != 'Unknown':
                print(f"📝 Found document with author data:")
                print(f"   author: {author_field}")
                print(f"   authors: {authors_field}")
                print(f"   title: {title}")
                break
        
        return True, es, list(fields)
        
    except Exception as e:
        print(f"❌ Debug failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, []


def test_field_variations(es, fields):
    """Test different field variations for author search."""
    print("\n🧪 Testing field variations...")
    
    test_author = "Anna Dubois"
    
    # Test different field names
    field_tests = []
    
    if 'author' in fields:
        field_tests.append('author')
    if 'authors' in fields:
        field_tests.append('authors')
    if 'Author' in fields:
        field_tests.append('Author')
    if 'authorname' in fields:
        field_tests.append('authorname')
    if 'author_name' in fields:
        field_tests.append('author_name')
    
    print(f"🔍 Testing author fields: {field_tests}")
    
    for field in field_tests:
        print(f"\n🔍 Testing field: {field}")
        
        # Test exact match
        exact_query = {
            "query": {
                "match_phrase": {
                    field: test_author
                }
            },
            "size": 5
        }
        
        try:
            response = es.search(index="research-publications-static", body=exact_query)
            hits = response['hits']['hits']
            print(f"   Exact match: {len(hits)} results")
            
            if hits:
                print(f"   Sample result: {hits[0]['_source'].get(field, 'No field')}")
                return True, field, hits
                
        except Exception as e:
            print(f"   ❌ Exact match failed: {str(e)}")
        
        # Test partial match
        partial_query = {
            "query": {
                "match": {
                    field: test_author
                }
            },
            "size": 5
        }
        
        try:
            response = es.search(index="research-publications-static", body=partial_query)
            hits = response['hits']['hits']
            print(f"   Partial match: {len(hits)} results")
            
            if hits:
                print(f"   Sample result: {hits[0]['_source'].get(field, 'No field')}")
                return True, field, hits
                
        except Exception as e:
            print(f"   ❌ Partial match failed: {str(e)}")
        
        # Test wildcard
        wildcard_query = {
            "query": {
                "wildcard": {
                    field: "*Dubois*"
                }
            },
            "size": 5
        }
        
        try:
            response = es.search(index="research-publications-static", body=wildcard_query)
            hits = response['hits']['hits']
            print(f"   Wildcard match: {len(hits)} results")
            
            if hits:
                print(f"   Sample result: {hits[0]['_source'].get(field, 'No field')}")
                return True, field, hits
                
        except Exception as e:
            print(f"   ❌ Wildcard match failed: {str(e)}")
    
    return False, None, []


def search_for_any_dubois(es, fields):
    """Search for any author with 'Dubois' in any field."""
    print("\n🔍 Searching for ANY author with 'Dubois'...")
    
    # Try multi-field search
    multi_query = {
        "query": {
            "multi_match": {
                "query": "Dubois",
                "fields": fields,
                "type": "phrase_prefix"
            }
        },
        "size": 10
    }
    
    try:
        response = es.search(index="research-publications-static", body=multi_query)
        hits = response['hits']['hits']
        print(f"✅ Multi-field search: {len(hits)} results")
        
        if hits:
            print(f"📋 Sample results:")
            for i, hit in enumerate(hits[:3], 1):
                source = hit['_source']
                print(f"   {i}. Fields with 'Dubois':")
                for field in fields:
                    value = source.get(field, '')
                    if value and 'Dubois' in str(value):
                        print(f"      {field}: {value}")
                print()
        
        return hits
        
    except Exception as e:
        print(f"❌ Multi-field search failed: {str(e)}")
        return []


def main():
    """Run all debug tests."""
    print("🔍 Debugging Search Tools - Finding Anna Dubois")
    print("=" * 60)
    
    # Debug 1: Database structure
    struct_success, es, fields = debug_database_structure()
    if not struct_success:
        print("❌ Cannot proceed without database structure")
        return 1
    
    # Debug 2: Field variations
    field_success, working_field, hits = test_field_variations(es, fields)
    
    # Debug 3: General search
    general_hits = search_for_any_dubois(es, fields)
    
    # Summary
    print("\n" + "=" * 60)
    print("🔍 Debug Results:")
    
    if field_success:
        print(f"✅ Found Anna Dubois in field: {working_field}")
        print(f"✅ Results: {len(hits)} publications")
        print(f"💡 The search_by_author function should use '{working_field}' not 'authors'")
    else:
        print("❌ Anna Dubois not found in any author field")
        
    if general_hits:
        print(f"✅ Found {len(general_hits)} results with 'Dubois' in any field")
        print("💡 Check if the author data is in a different field")
    else:
        print("❌ No 'Dubois' found in any field")
        print("💡 The author might be stored differently or data quality issue")
    
    print(f"\n📋 Available fields: {fields}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())