import React from "react";
import "./App.css";

function App() {
  return (
    <div className="page">
      {/* Navbar */}
      <nav className="nav">
        <h1 className="logo">AVASTHI</h1>
        <ul className="nav-links">
          <li>HOME</li>
          <li>ABOUT US</li>
          <li>RESOURCES</li>
          <li>CONTACT</li>
        </ul>
      </nav>

      {/* Hero Section */}
      <section className="hero">
        <div className="hero-text">
          <p><span>Aspire</span> to</p>
          <p><span>Vibrance</span>, embrace</p>
          <p><span>Adaptability</span> with strength</p>
          <p><span>Support</span> balance,</p>
          <p><span>Tranquility</span>, compassion</p>
          <p><span>Heal</span> through hope and instill</p>
          <p><span>Independence</span></p>
        </div>

        <div className="hero-img">
          {/* using public/meditation.png -> path starts with / */}
          <img src="/meditation.png" alt="Meditation" />
        </div>
      </section>

      {/* First Button Row */}
      <section className="button-row">
  <div className="feature-card">
    <img src="/chat.png" alt="Chat" />
    <h3>Chat with Me!</h3>
    <p>Instantly connect for emotional support and guidance.</p>
  </div>

  <div className="feature-card">
    <img src="/tips.png" alt="Tips" />
    <h3>Get Tips</h3>
    <p>Receive personalized mental health tips and practices.</p>
  </div>

  <div className="feature-card">
    <img src="/diet.png" alt="Healthy Diet" />
    <h3>Healthy Diet</h3>
    <p>Explore diet plans that boost mental and physical wellness.</p>
  </div>
</section>

      {/* Explore Resources */}
      {/* Explore Resources */}
<section className="resources">
  <h2>Explore Resources</h2>
  <div className="resource-grid">
    <div className="resource-card">
      <img src="/cbt.png" alt="CBT" />
      <h3>CBT</h3>
      <p>Practice cognitive behavioral therapy techniques.</p>
    </div>
    <div className="resource-card">
      <img src="/meditation1.png" alt="Guided Meditation" />
      <h3>Guided Meditation</h3>
      <p>Relax your mind with calming meditation sessions.</p>
    </div>
    <div className="resource-card">
      <img src="/music.png" alt="Music Therapy" />
      <h3>Music Therapy</h3>
      <p>Heal with soothing and uplifting music therapy.</p>
    </div>
    <div className="resource-card">
      <img src="/journal.png" alt="Journal" />
      <h3>Journal</h3>
      <p>Express thoughts and track your emotional progress.</p>
    </div>
  </div>
</section>

    </div>
  );
}

export default App;
