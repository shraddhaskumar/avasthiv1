// src/App.jsx
import React from "react";

export default function App() {
  return (
    <div className="bg-[#F5F5F5] min-h-screen font-poppins">
      {/* Navbar */}
      <nav className="flex justify-between items-center px-10 py-4 bg-purple-100 shadow-md">
        <h1 className="font-bold text-lg tracking-wide">AVASTHI</h1>
        <ul className="flex gap-8 text-sm font-medium">
          <li className="hover:text-purple-700 cursor-pointer">HOME</li>
          <li className="hover:text-purple-700 cursor-pointer">ABOUT US</li>
          <li className="hover:text-purple-700 cursor-pointer">RESOURCES</li>
          <li className="hover:text-purple-700 cursor-pointer">CONTACT</li>
        </ul>
      </nav>

      {/* Hero Section */}
      <section className="flex flex-col md:flex-row items-center justify-between px-10 py-12">
        {/* Left Text */}
        <div className="space-y-2 text-lg">
          <p>
            <span className="text-purple-700 font-semibold">Aspire</span> to
          </p>
          <p>
            <span className="text-purple-700 font-semibold">Vibrance</span>,
            embrace
          </p>
          <p>
            <span className="text-purple-700 font-semibold">Adaptability</span>{" "}
            with strength
          </p>
          <p>
            <span className="text-purple-700 font-semibold">Support</span>{" "}
            balance,
          </p>
          <p>
            <span className="text-purple-700 font-semibold">Tranquility</span>,
            compassion
          </p>
          <p>
            <span className="text-purple-700 font-semibold">Heal</span> through
            hope and instill
          </p>
          <p>
            <span className="text-purple-700 font-semibold">Independence</span>
          </p>
        </div>

        {/* Right Image */}
        <div className="mt-6 md:mt-0">
          <img
            src="/meditation.png"
            alt="Meditation"
            className="w-72 h-auto rounded-2xl shadow-md"
          />
        </div>
      </section>

      {/* First Button Row */}
      <section className="flex justify-center gap-6 py-6">
        <button className="bg-purple-200 px-6 py-2 rounded-full shadow hover:bg-purple-300">
          Chat with Me!
        </button>
        <button className="bg-purple-200 px-6 py-2 rounded-full shadow hover:bg-purple-300">
          Get Tips
        </button>
        <button className="bg-purple-200 px-6 py-2 rounded-full shadow hover:bg-purple-300">
          Healthy Diet
        </button>
      </section>

      {/* Explore Resources */}
      <section className="px-10 py-6">
        <h2 className="text-xl font-semibold mb-4">Explore Resources</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-purple-100 py-4 px-6 rounded-md text-center shadow">
            CBT
          </div>
          <div className="bg-purple-100 py-4 px-6 rounded-md text-center shadow">
            Guided Meditation
          </div>
          <div className="bg-purple-100 py-4 px-6 rounded-md text-center shadow">
            Music Therapy
          </div>
          <div className="bg-purple-100 py-4 px-6 rounded-md text-center shadow">
            Journal
          </div>
        </div>
      </section>
    </div>
  );
}
