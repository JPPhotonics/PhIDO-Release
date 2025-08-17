import streamlit as st

# Example content for different sections
section1_content = """
Why it matters: The landmark 6-3 ruling along ideological lines overturns the court's 40-year-old "Chevron deference" doctrine. It could make it harder for executive agencies to tackle a wide array of policy areas, including environmental and health regulations and labor and employment laws.
"""

section2_content = """
This is the content of section 2. It can also contain multiple lines of text or other elements.
"""

# Custom JavaScript to toggle sections
script = """
<script>
function toggleSection(sectionId) {
    var section = document.getElementById(sectionId);
    section.classList.toggle("collapsed");
}
</script>
"""

# Render sections with custom JavaScript
st.markdown(script, unsafe_allow_html=True)

with st.expander(section1_content[:300], expanded=False):
    st.markdown(
        f'<div id="section1" class="collapsed">{section1_content}</div>',
        unsafe_allow_html=True,
    )

with st.expander("Section 2", expanded=False):
    st.markdown(
        f'<div id="section2" class="collapsed">{section2_content}</div>',
        unsafe_allow_html=True,
    )
