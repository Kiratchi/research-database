#!/usr/bin/env python3
"""
Standalone script to test Elasticsearch mapping and field structure.
Run this from your project root directory.
"""

import json
import os
from elasticsearch import Elasticsearch

def main():
    print("🔍 ELASTICSEARCH MAPPING TEST")
    print("="*60)
    
    # Load environment variables (adjust paths as needed)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("⚠️ python-dotenv not available, using os.environ directly")
    
    # Initialize Elasticsearch client
    try:
        es_client = Elasticsearch(
            [os.getenv('ES_HOST', 'localhost:9200')],
            http_auth=(
                os.getenv('ES_USER', 'elastic'),
                os.getenv('ES_PASS', 'password')
            ),
            verify_certs=False,
            ssl_show_warn=False
        )
        
        # Test connection
        if es_client.ping():
            print("✅ Elasticsearch connection successful")
        else:
            print("❌ Elasticsearch connection failed")
            return
            
    except Exception as e:
        print(f"❌ Failed to connect to Elasticsearch: {e}")
        return
    
    index_name = "research-publications-static"
    
    # TEST 1: Get index mapping
    print(f"\n📋 STEP 1: GET MAPPING FOR INDEX '{index_name}'")
    print("-"*50)
    
    try:
        mapping_response = es_client.indices.get_mapping(index=index_name)
        mapping = mapping_response[index_name]['mappings']
        
        print("✅ Mapping retrieved successfully")
        
        # Save full mapping to file for detailed inspection
        with open('elasticsearch_mapping.json', 'w') as f:
            json.dump(mapping, f, indent=2)
        print("💾 Full mapping saved to 'elasticsearch_mapping.json'")
        
        # Print relevant fields
        if 'properties' in mapping:
            properties = mapping['properties']
            print(f"\n🔍 FOUND {len(properties)} TOP-LEVEL FIELDS:")
            for field_name in sorted(properties.keys()):
                field_info = properties[field_name]
                field_type = field_info.get('type', 'object')
                print(f"  - {field_name}: {field_type}")
            
            # Focus on Persons field
            if 'Persons' in properties:
                print(f"\n🎯 PERSONS FIELD STRUCTURE:")
                persons_field = properties['Persons']
                print(json.dumps(persons_field, indent=2))
            else:
                print("\n❌ No 'Persons' field found in mapping")
        
    except Exception as e:
        print(f"❌ Error getting mapping: {e}")
        return
    
    # TEST 2: Get sample documents
    print(f"\n📋 STEP 2: GET SAMPLE DOCUMENTS")
    print("-"*50)
    
    try:
        sample_query = {
            "size": 3,
            "query": {"match_all": {}}
        }
        
        response = es_client.search(index=index_name, body=sample_query)
        print(f"✅ Retrieved {len(response['hits']['hits'])} sample documents")
        
        # Analyze first document structure
        if response['hits']['hits']:
            first_doc = response['hits']['hits'][0]['_source']
            
            print(f"\n🔍 FIRST DOCUMENT FIELDS:")
            for field_name in sorted(first_doc.keys()):
                field_value = first_doc[field_name]
                field_type = type(field_value).__name__
                print(f"  - {field_name}: {field_type}")
            
            # Focus on Persons field in actual data
            if 'Persons' in first_doc:
                print(f"\n🎯 PERSONS FIELD IN ACTUAL DATA:")
                persons_data = first_doc['Persons']
                print(f"Type: {type(persons_data)}")
                print(f"Structure:")
                print(json.dumps(persons_data, indent=2))
                
                # Save sample documents
                with open('sample_documents.json', 'w') as f:
                    json.dump(response['hits']['hits'], f, indent=2)
                print("💾 Sample documents saved to 'sample_documents.json'")
            else:
                print("\n❌ No 'Persons' field in sample document")
    
    except Exception as e:
        print(f"❌ Error getting sample documents: {e}")
        return
    
    # TEST 3: Test different query approaches
    print(f"\n📋 STEP 3: TEST QUERY APPROACHES")
    print("-"*50)
    
    test_queries = [
        {
            "name": "Simple terms aggregation on DisplayName.keyword",
            "query": {
                "size": 0,
                "aggs": {
                    "authors": {
                        "terms": {
                            "field": "Persons.PersonData.DisplayName.keyword",
                            "size": 5
                        }
                    }
                }
            }
        },
        {
            "name": "Simple terms aggregation on DisplayName (no .keyword)",
            "query": {
                "size": 0,
                "aggs": {
                    "authors": {
                        "terms": {
                            "field": "Persons.PersonData.DisplayName",
                            "size": 5
                        }
                    }
                }
            }
        },
        {
            "name": "Nested aggregation on Persons",
            "query": {
                "size": 0,
                "aggs": {
                    "nested_authors": {
                        "nested": {"path": "Persons"},
                        "aggs": {
                            "authors": {
                                "terms": {
                                    "field": "Persons.PersonData.DisplayName.keyword",
                                    "size": 5
                                }
                            }
                        }
                    }
                }
            }
        },
        {
            "name": "Nested aggregation on Persons.PersonData",
            "query": {
                "size": 0,
                "aggs": {
                    "nested_authors": {
                        "nested": {"path": "Persons.PersonData"},
                        "aggs": {
                            "authors": {
                                "terms": {
                                    "field": "Persons.PersonData.DisplayName.keyword",
                                    "size": 5
                                }
                            }
                        }
                    }
                }
            }
        },
        {
            "name": "Match query on DisplayName (like search_by_author)",
            "query": {
                "size": 3,
                "query": {
                    "match": {
                        "Persons.PersonData.DisplayName": "Adrian"
                    }
                }
            }
        },
        {
            "name": "Nested query on Persons.PersonData (like count_entities attempt)",
            "query": {
                "size": 3,
                "query": {
                    "nested": {
                        "path": "Persons.PersonData",
                        "query": {
                            "match": {
                                "Persons.PersonData.DisplayName": "Adrian"
                            }
                        }
                    }
                }
            }
        },
        # NEW TESTS - Testing the proposed fix
        {
            "name": "PROPOSED FIX: Simple match + terms aggregation (Adrian count)",
            "query": {
                "size": 0,
                "query": {
                    "match": {
                        "Persons.PersonData.DisplayName": "Adrian"
                    }
                },
                "aggs": {
                    "unique_authors": {
                        "terms": {
                            "field": "Persons.PersonData.DisplayName.keyword",
                            "size": 50,
                            "include": ".*[Aa]drian.*"
                        }
                    },
                    "total_publications": {
                        "value_count": {"field": "_id"}
                    }
                }
            }
        },
        {
            "name": "PROPOSED FIX: Alternative with regex include pattern",
            "query": {
                "size": 0,
                "query": {
                    "match": {
                        "Persons.PersonData.DisplayName": "Adrian"
                    }
                },
                "aggs": {
                    "unique_authors": {
                        "terms": {
                            "field": "Persons.PersonData.DisplayName.keyword",
                            "size": 100
                        }
                    },
                    "total_publications": {
                        "value_count": {"field": "_id"}
                    }
                }
            }
        },
        {
            "name": "VERIFY: Get sample Adrian authors from aggregation",
            "query": {
                "size": 0,
                "aggs": {
                    "all_authors": {
                        "terms": {
                            "field": "Persons.PersonData.DisplayName.keyword",
                            "size": 1000,
                            "include": ".*Adrian.*"
                        }
                    }
                }
            }
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n🧪 TEST {i}: {test['name']}")
        try:
            result = es_client.search(index=index_name, body=test['query'])
            print(f"✅ SUCCESS - Query worked!")
            
            if 'aggregations' in result:
                print("📊 Aggregation results:")
                for agg_name, agg_data in result['aggregations'].items():
                    if agg_name == "unique_authors" and "buckets" in agg_data:
                        # Special handling for Adrian counting tests
                        buckets = agg_data["buckets"]
                        print(f"  {agg_name}: Found {len(buckets)} unique authors")
                        
                        # Count how many actually contain "Adrian"
                        adrian_count = 0
                        adrian_names = []
                        for bucket in buckets:
                            name = bucket["key"]
                            count = bucket["doc_count"]
                            if "adrian" in name.lower():
                                adrian_count += 1
                                adrian_names.append(f"{name} ({count} pubs)")
                        
                        print(f"  📈 Authors with 'Adrian': {adrian_count}")
                        if adrian_names:
                            print(f"  👥 Names: {', '.join(adrian_names[:10])}")
                            if len(adrian_names) > 10:
                                print(f"      ... and {len(adrian_names) - 10} more")
                    
                    elif agg_name == "total_publications":
                        print(f"  {agg_name}: {agg_data.get('value', 'N/A')} publications")
                    
                    else:
                        # Standard aggregation display
                        print(f"  {agg_name}: {json.dumps(agg_data, indent=2)[:200]}...")
            
            if 'hits' in result and result['hits']['hits']:
                total_hits = result['hits']['total']
                if isinstance(total_hits, dict):
                    total_hits = total_hits.get('value', 0)
                print(f"📄 Found {total_hits} total hits")
                print(f"🔍 First result preview: {str(result['hits']['hits'][0])[:200]}...")
                
        except Exception as e:
            print(f"❌ FAILED - {str(e)[:150]}...")
            
        # Add separator for readability
        if i in [6, 9]:  # After original tests and after new tests
            print("\n" + "="*60)
    
    print(f"\n🎯 SUMMARY")
    print("="*60)
    print("Check the generated files:")
    print("  - elasticsearch_mapping.json (full mapping)")  
    print("  - sample_documents.json (sample data)")
    print("\nLook for which query approaches succeeded vs failed!")

if __name__ == "__main__":
    main()