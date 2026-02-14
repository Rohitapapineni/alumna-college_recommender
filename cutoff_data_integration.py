def search_current_cutoffs(self, preferences: UserPreferences, year: int = 2025) -> Dict:
    """Search for current year cutoff data"""
    try:
        query = f"JEE Main cutoff {year} {preferences.location or 'India'}"
        if preferences.specific_institution_type:
            query = f"{preferences.specific_institution_type} cutoff {year} {preferences.state or 'India'}"
        
        results = self.tavily_client.search(
            query=query,
            max_results=3,
            search_depth="advanced"
        )
        
        # Parse cutoff information
        cutoff_data = self._extract_cutoff_data(results)
        return cutoff_data
        
    except Exception as e:
        logger.error(f"Error searching cutoffs: {e}")
        return {}
    
    def _extract_cutoff_data(self, search_results: Dict) -> Dict:
        """Extract structured cutoff data from search results"""
        # Use LLM to parse cutoff information
        results_text = json.dumps([
            {"title": r['title'], "content": r['content'][:300]}
            for r in search_results.get('results', [])
        ])
        
        extraction_prompt = f"""Extract cutoff information from these search results:

{results_text}

Return as JSON: {{
    "iit_cutoff_percentile": number,
    "nit_cutoff_percentile": number,
    "private_college_cutoff": number,
    "year": 2025,
    "source": "web_search"
}}"""
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.1,
            max_tokens=300
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {}
