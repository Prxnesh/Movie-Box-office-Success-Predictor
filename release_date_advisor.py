# release_date_advisor.py
import streamlit as st

def suggest_release_date(genre):
    """
    Suggest ideal release windows based on genre seasonality.
    """
    st.markdown(f"### 📅 Ideal Release Windows for {genre}")
    
    seasonality = {
        'Action': {
            'Best': ['May', 'July', 'December'],
            'Reason': "Summer blockbusters and holiday season maximize turnout."
        },
        'Sci-Fi': {
            'Best': ['May', 'June', 'November'],
            'Reason': "Strong summer appeal; pre-holiday slots work well for epics."
        },
        'Horror': {
            'Best': ['October', 'January', 'September'],
            'Reason': "Halloween (Oct) is peak. Jan/Sept are low-competition months where horror thrives."
        },
        'Comedy': {
            'Best': ['February', 'July', 'August'],
            'Reason': "Valentines (Feb) or mid-summer counter-programming."
        },
        'Drama': {
            'Best': ['October', 'November', 'December'],
            'Reason': "Awards season window. Best for prestige films."
        },
        'Romance': {
            'Best': ['February', 'December'],
            'Reason': "Valentines Day and Holiday season."
        },
        'Animation': {
            'Best': ['June', 'November', 'December'],
            'Reason': "School holidays and family gathering times."
        },
        'Thriller': {
            'Best': ['March', 'August', 'October'],
            'Reason': "Shoulder seasons where adult audiences are available."
        }
    }
    
    advice = seasonality.get(genre, {
        'Best': ['March', 'April', 'September'],
        'Reason': "Shoulder months offer lower competition for niche genres."
    })
    
    cols = st.columns(len(advice['Best']))
    for i, month in enumerate(advice['Best']):
        with cols[i]:
            st.markdown(f"""
            <div style="background-color: #262730; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #4B4B4B;">
                <h4 style="margin:0; color: #FF4B4B;">{month}</h4>
            </div>
            """, unsafe_allow_html=True)
            
    st.info(f"💡 **Why?** {advice['Reason']}")
    
    st.markdown("#### ⚔️ Competitive Clash Analyzer")
    st.caption("Simulated analysis of potential clashes")
    
    clash_risk = "High" if genre in ['Action', 'Sci-Fi'] else "Medium"
    st.metric("Competition Risk", clash_risk, delta="Avoid Major Franchises" if clash_risk=="High" else "Safe", delta_color="inverse")
