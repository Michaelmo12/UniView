import "./Home.css";

const DRONE_IDS = [1, 2, 3, 4, 5, 6, 7, 8];

function DroneCell({ id }: { id: number }) {
  const isLive = false; // wired to stream data in future

  return (
    <div className={`drone-cell ${isLive ? "drone-cell--live" : "drone-cell--offline"}`}>
      <div className="drone-cell__scanlines" />
      <div className="drone-cell__corners" />

      <div className="drone-cell__header">
        <span className="drone-cell__id">DRONE-{String(id).padStart(2, "0")}</span>
        <span className={`drone-cell__status ${isLive ? "drone-cell__status--live" : ""}`}>
          {isLive ? (
            <><span className="drone-cell__rec-dot" />REC</>
          ) : (
            "NO SIGNAL"
          )}
        </span>
      </div>

      <div className="drone-cell__body">
        {!isLive && (
          <div className="drone-cell__nosignal">
            <div className="drone-cell__crosshair" />
          </div>
        )}
      </div>

      <div className="drone-cell__footer">
        <span>1920×1080</span>
        <span>CH-{String(id).padStart(2, "0")}</span>
        <span>30FPS</span>
      </div>
    </div>
  );
}

function Home() {
  return (
    <div className="uniview-dashboard">
      <main className="uniview-grid-wrapper">
        <div className="uniview-grid">
          {DRONE_IDS.map((id) => (
            <DroneCell key={id} id={id} />
          ))}
        </div>
      </main>
    </div>
  );
}

export default Home;
