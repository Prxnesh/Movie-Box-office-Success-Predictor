# budget_optimizer.py
import streamlit as st

def optimize_budget(budget, rating, genre, predicted_collection_range):
    """
    Analyze budget vs potential and suggest optimizations.
    """
    suggestions = []
    
    # Parse collection range (very rough heuristic)
    # Ranges: <1M, 1M-10M, 10M-50M, 50M-100M, 100M-300M, >300M
    lower_bound = 0
    if '>' in predicted_collection_range:
        lower_bound = float(predicted_collection_range.replace('>$', '').replace('M', '')) * 1e6
    elif '-' in predicted_collection_range:
        parts = predicted_collection_range.replace('$', '').replace('M', '').split('-')
        lower_bound = float(parts[0]) * 1e6
        
    roi = lower_bound / budget if budget > 0 else 0
    
    st.markdown("### 💰 Budget ROI Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Budget", f"${budget/1e6:.1f}M")
    with col2:
        st.metric("Est. Base ROI", f"{roi:.1f}x", delta="Good" if roi > 2.0 else "Risk", delta_color="normal" if roi > 2.0 else "inverse")
        
    st.markdown("### 💡 Optimization Suggestions")
    
    if roi < 1.5:
        st.warning("⚠️ **High Risk Detected**: Projected returns are low relative to budget.")
        if budget > 150e6:
            suggestions.append({
                "action": "Cut VFX/Production Costs",
                "impact": "High",
                "detail": "Budget is in Ultra tier but ROI is low. Consider reducing VFX shots or filming locations."
            })
        if rating < 7.0:
            suggestions.append({
                "action": "Improve Script Quality",
                "impact": "Critical",
                "detail": "Low rating is dragging down potential. Invest in script doctoring rather than production."
            })
    elif roi > 3.0:
        st.success("🚀 **High Potential**: Strong ROI expected.")
        if budget < 50e6:
            suggestions.append({
                "action": "Increase Marketing Spend",
                "impact": "Medium",
                "detail": "You have a strong product. Aggressive marketing could push it to the next tier."
            })
            
    if genre in ['Action', 'Sci-Fi'] and budget < 80e6:
        suggestions.append({
            "action": "Review VFX Budget",
            "impact": "High",
            "detail": f"{genre} movies typically require higher budgets for competitive visuals."
        })
        
    if not suggestions:
        st.info("✅ Current budget allocation looks healthy for this genre and rating.")
    else:
        for s in suggestions:
            with st.expander(f"🔧 {s['action']} ({s['impact']} Impact)"):
                st.write(s['detail'])
                
    return suggestions
