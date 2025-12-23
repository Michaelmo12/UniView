function Home() {
  // דוגמה לנתוני סטרימים - אתה תוכל להחליף את זה בנתונים אמיתיים מה-API
  const mockStreams = [
    { id: 1, title: 'Camera 001', detections: 3, isActive: true, color: 'blue' },
    { id: 2, title: 'Camera 002', detections: 1, isActive: true, color: 'green' },
    { id: 3, title: 'Camera 003', detections: 0, isActive: true, color: 'orange' },
    { id: 4, title: 'Camera 004', detections: 5, isActive: true, color: 'purple' },
    { id: 5, title: 'Camera 005', detections: 2, isActive: false, color: 'blue' },
    { id: 6, title: 'Camera 006', detections: 0, isActive: true, color: 'yellow' },
  ];

  return (
    <div className="page-content">
      <h1>UniView - Surveillance Dashboard</h1>
      <p>Real-time monitoring and tracking system</p>

      {/* Grid של קופסאות NEON לסטרימים */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
        gap: '2rem',
        marginTop: '2rem'
      }}>
        {mockStreams.map((stream) => (
          <div key={stream.id} className={`neon-box ${stream.color}`}>
            {/* Detection Count Label */}
            <div style={{ marginBottom: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className={`status-label ${stream.detections > 0 ? 'tracking' : 'idle'}`}>
                {stream.detections > 0 ? `${stream.detections} TARGETS` : 'NO TARGETS'}
              </span>
              <span style={{
                fontSize: '0.7rem',
                color: stream.isActive ? 'var(--neon-green)' : '#ff0000',
                fontWeight: '600',
                letterSpacing: '0.5px'
              }}>
                {stream.isActive ? 'ACTIVE' : 'INACTIVE'}
              </span>
            </div>

            {/* Camera Title */}
            <h3 style={{
              color: 'var(--neon-blue)',
              fontSize: '1.25rem',
              marginBottom: '0.25rem',
              textShadow: '0 0 10px currentColor'
            }}>
              {stream.title}
            </h3>

            {/* Stream Info */}
            <div style={{
              color: '#cbd5e0',
              fontSize: '0.875rem',
              marginTop: '1rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.5rem'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Resolution:</span>
                <span style={{ color: 'var(--neon-green)' }}>1920x1080</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>FPS:</span>
                <span style={{ color: 'var(--neon-green)' }}>30</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Status:</span>
                <span style={{ color: stream.isActive ? 'var(--neon-green)' : '#ff0000' }}>
                  {stream.isActive ? 'Active' : 'Inactive'}
                </span>
              </div>
            </div>

            {/* View Button */}
            <button style={{
              marginTop: '1rem',
              width: '100%',
              padding: '0.75rem',
              background: 'rgba(0, 212, 255, 0.1)',
              border: '1px solid var(--neon-blue)',
              borderRadius: '6px',
              color: 'var(--neon-blue)',
              fontSize: '0.875rem',
              fontWeight: '600',
              letterSpacing: '1px',
              textTransform: 'uppercase',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              boxShadow: '0 0 10px rgba(0, 212, 255, 0.2)',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = 'rgba(0, 212, 255, 0.2)';
              e.currentTarget.style.boxShadow = '0 0 20px rgba(0, 212, 255, 0.5), 0 0 40px rgba(0, 212, 255, 0.3)';
              e.currentTarget.style.transform = 'translateY(-2px)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'rgba(0, 212, 255, 0.1)';
              e.currentTarget.style.boxShadow = '0 0 10px rgba(0, 212, 255, 0.2)';
              e.currentTarget.style.transform = 'translateY(0)';
            }}>
              View Stream
            </button>
          </div>
        ))}
      </div>

      {/* הערה למשתמש */}
      <div style={{
        marginTop: '3rem',
        padding: '1.5rem',
        background: 'rgba(0, 212, 255, 0.05)',
        border: '1px solid rgba(0, 212, 255, 0.2)',
        borderRadius: '8px',
        color: '#cbd5e0'
      }}>
        <p style={{ margin: 0, fontSize: '0.875rem' }}>
          💡 <strong style={{ color: 'var(--neon-blue)' }}>Demo Mode:</strong> זהו דאשבורד דמו למעקב אחר אובייקטים.
          כל קופסה מייצגת מצלמה עם מספר האובייקטים שזוהו (TARGETS).
          כאשר תחבר את ה-API האמיתי, הנתונים יתעדכנו בזמן אמת עם מספר האנשים/אובייקטים שמזוהים בכל סטרים.
        </p>
      </div>
    </div>
  );
}

export default Home;
