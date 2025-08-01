"""Persons search tool implementation."""
from typing import Dict, Any, List, Optional, Type

from ..base.elasticsearch_tool import BaseElasticsearchSearchTool
from .models import PersonsSearchInput, PersonStandard, PersonExpanded, PersonFull


class PersonsSearchTool(BaseElasticsearchSearchTool):
    """Search tool for researcher profiles."""
    
    name: str = "search_persons"
    description: str = """Search for researchers/persons in the Chalmers academic database BY NAME ONLY.
    
    CRITICAL: This tool searches ONLY by person names, NOT by research topics or fields!
    
    WHEN TO USE THIS TOOL:
    ✓ You know a specific person's name (e.g., "Find John Doe")
    ✓ You want all researchers in a department (e.g., organization="Physics")
    ✓ You need researcher identifiers (ORCID, Scopus ID)
    
    WHEN NOT TO USE THIS TOOL:
    ✗ Finding "researchers in quantum computing" → Use search_publications first, then extract author names
    ✗ Finding "leading AI researchers" → Use search_publications to find prolific authors
    ✗ Any query about research topics → This tool CANNOT search by research field
    
    CORRECT USAGE:
    - search_persons(query="John Doe") → Find specific person
    - search_persons(query="", organization="Computer Science") → All CS researchers
    - search_persons(query="Smith", has_publications=true) → All Smiths with publications
    
    INCORRECT USAGE:
    - search_persons(query="machine learning") → Will return 0 results
    - search_persons(query="quantum computing") → Will return 0 results
    
    BEST PRACTICES:
    - Start with max_results=5-10 for initial searches
    - Only use field_selection="expanded" if you specifically need ORCID, Scopus ID, or organization details
    
    RETURNS: List of persons, each containing:
    - Standard: (id, display_name, first_name, last_name, is_active, has_publications, has_orcid, score)
    - Expanded: + (email, phone, identifiers(orcid, scopus_id), organizations[(name, unit)], publication_count)
    - Full: + (all contact info, all_identifiers, all_organizations, detailed_profile)
    """
    
    args_schema: Type[PersonsSearchInput] = PersonsSearchInput
    
    def _build_search_request(
        self,
        query: str,
        max_results: int,
        offset: int,
        sort_by: str,
        field_selection: str,
        filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build Elasticsearch query for persons."""
        
        # Start with base query structure
        es_query = {
            "size": max_results,
            "from": offset
        }
        
        # Build query clauses
        must_clauses = []
        filter_clauses = []
        
        # Main query - search in name fields
        if query:
            must_clauses.append({
                "multi_match": {
                    "query": query,
                    "fields": [
                        "DisplayName^3",
                        "FirstName^2", 
                        "LastName^2",
                        "DisplayName.keyword"
                    ],
                    "type": "best_fields",
                    "operator": "and"
                }
            })
        
        # Apply filters
        if filters.get('active_only', True):
            filter_clauses.append({"term": {"IsActive": True}})
        
        if filters.get('organization'):
            # Search in organization names - ES 6.x doesn't support nested queries well
            # Use a simpler approach
            filter_clauses.append({
                "bool": {
                    "should": [
                        {"match": {"OrganizationHome.DisplayNameEng": filters['organization']}},
                        {"match": {"OrganizationHome.DisplayNameSwe": filters['organization']}},
                        {"match": {"OrganizationHome.NameEng": filters['organization']}},
                        {"match": {"OrganizationHome.NameSwe": filters['organization']}}
                    ],
                    "minimum_should_match": 1
                }
            })
        
        if filters.get('has_orcid') is not None:
            if filters['has_orcid']:
                filter_clauses.append({"exists": {"field": "IdentifierOrcid"}})
            else:
                filter_clauses.append({"bool": {"must_not": {"exists": {"field": "IdentifierOrcid"}}}})
        
        if filters.get('has_publications') is not None:
            filter_clauses.append({"term": {"HasPublications": filters['has_publications']}})
        
        # Always exclude deleted records
        filter_clauses.append({"term": {"IsDeleted": False}})
        
        # Construct the bool query
        bool_query = {}
        if must_clauses:
            bool_query["must"] = must_clauses
        if filter_clauses:
            bool_query["filter"] = filter_clauses
        
        # Set the query
        if bool_query:
            es_query["query"] = {"bool": bool_query}
        else:
            es_query["query"] = {"match_all": {}}
        
        # Add sorting
        es_query["sort"] = self._build_sort_clause(sort_by)
        
        # Add source filtering for optimization
        source_fields = self._get_source_fields(field_selection)
        if source_fields:
            es_query["_source"] = source_fields
        
        return es_query
    
    def _build_sort_clause(self, sort_by: str) -> List[Dict[str, Any]]:
        """Build sort clause specific to persons."""
        if sort_by == "relevance":
            return ["_score", {"DisplayName.keyword": "asc"}]
        elif sort_by == "name_asc":
            return [{"DisplayName.keyword": "asc"}]
        elif sort_by == "date_desc":
            # Sort by creation date if available
            return [{"CreatedAt": "desc"}, {"DisplayName.keyword": "asc"}]
        elif sort_by == "date_asc":
            return [{"CreatedAt": "asc"}, {"DisplayName.keyword": "asc"}]
        else:
            return ["_score"]
    
    def _get_source_fields(self, field_selection: str) -> Optional[List[str]]:
        """Get source fields to retrieve based on selection level."""
        if field_selection == "standard":
            return [
                "Id", "DisplayName", "FirstName", "LastName",
                "IsActive", "HasPublications", "HasProjects",
                "IdentifierOrcid", "HasIdentifiers"
            ]
        elif field_selection == "expanded":
            return [
                "Id", "DisplayName", "FirstName", "LastName",
                "IsActive", "HasPublications", "HasProjects",
                "IdentifierOrcid", "IdentifierCid", "Identifiers",
                "OrganizationHome", "OrganizationHomeCount"
            ]
        else:
            # Full - return everything except maintenance fields
            return None
    
    def _transform_results(
        self,
        raw_results: Dict[str, Any],
        field_selection: str
    ) -> List[Dict[str, Any]]:
        """Transform raw results based on field selection."""
        
        transformed = []
        
        for hit in raw_results['hits']['hits']:
            doc = hit.get('_source')
            if not doc:
                continue
            score = hit.get('_score')
            
            if field_selection == "standard":
                # Minimal fields
                person = PersonStandard(
                    id=doc['Id'],
                    display_name=doc['DisplayName'],
                    first_name=doc.get('FirstName'),
                    last_name=doc.get('LastName'),
                    is_active=doc.get('IsActive', False),
                    has_publications=doc.get('HasPublications', False),
                    has_orcid=bool(doc.get('IdentifierOrcid')),
                    score=score
                )
                transformed.append(person.model_dump())
                
            elif field_selection == "expanded":
                # Include identifiers and affiliations
                # Extract ORCID
                orcid = None
                if doc.get('IdentifierOrcid'):
                    orcid = doc['IdentifierOrcid'][0] if isinstance(doc['IdentifierOrcid'], list) else doc['IdentifierOrcid']
                
                # Extract Scopus ID from Identifiers array
                scopus_id = self._extract_identifier(doc.get('Identifiers', []), 'ScopusAuthorId')
                
                # Extract organization names
                org_names = []
                if doc.get('OrganizationHome'):
                    for org in doc['OrganizationHome']:
                        if isinstance(org, dict):
                            name = org.get('DisplayNameEng') or org.get('DisplayNameSwe') or org.get('NameEng')
                            if name:
                                org_names.append(name)
                
                person = PersonExpanded(
                    id=doc['Id'],
                    display_name=doc['DisplayName'],
                    first_name=doc.get('FirstName'),
                    last_name=doc.get('LastName'),
                    is_active=doc.get('IsActive', False),
                    has_publications=doc.get('HasPublications', False),
                    has_orcid=bool(orcid),
                    score=score,
                    orcid=orcid,
                    scopus_id=scopus_id,
                    organization_names=org_names[:5],  # Limit to 5
                    organization_count=doc.get('OrganizationHomeCount', 0),
                    has_projects=doc.get('HasProjects', False)
                )
                transformed.append(person.model_dump())
                
            else:  # full
                # Return most fields but clean up maintenance fields
                full_doc = {
                    'id': doc['Id'],
                    'display_name': doc['DisplayName'],
                    'first_name': doc.get('FirstName'),
                    'last_name': doc.get('LastName'),
                    'is_active': doc.get('IsActive', False),
                    'has_publications': doc.get('HasPublications', False),
                    'has_projects': doc.get('HasProjects', False),
                    'has_orcid': bool(doc.get('IdentifierOrcid')),
                    'score': score,
                    'birth_year': doc.get('BirthYear', 0) if doc.get('BirthYear', 0) > 0 else None,
                    'identifiers': doc.get('Identifiers', []),
                    'organizations': doc.get('OrganizationHome', []),
                    'pdb_categories': doc.get('PdbCategories', []),
                }
                
                # Add all identifier arrays
                if doc.get('IdentifierOrcid'):
                    full_doc['orcid'] = doc['IdentifierOrcid'][0] if isinstance(doc['IdentifierOrcid'], list) else doc['IdentifierOrcid']
                if doc.get('IdentifierCid'):
                    full_doc['chalmers_id'] = doc['IdentifierCid']
                
                transformed.append(full_doc)
        
        return transformed
    
    def _extract_identifier(self, identifiers: List[Dict], id_type: str) -> Optional[str]:
        """Extract a specific identifier type from the identifiers array."""
        for identifier in identifiers:
            if identifier.get('Type') == id_type and identifier.get('IsActive', True):
                return identifier.get('Value')
        return None