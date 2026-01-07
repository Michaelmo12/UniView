import { PageHeader, Grid, StreamCard, InfoBox } from "../components/common";

function Home() {
  // Example stream data - you can replace this with real API data
  const mockStreams = [
    { id: 1, title: 'Camera 001', detections: 3, isActive: true, color: 'blue' as const },
    { id: 2, title: 'Camera 002', detections: 1, isActive: true, color: 'green' as const },
    { id: 3, title: 'Camera 003', detections: 0, isActive: true, color: 'orange' as const },
    { id: 4, title: 'Camera 004', detections: 5, isActive: true, color: 'purple' as const },
    { id: 5, title: 'Camera 005', detections: 2, isActive: false, color: 'blue' as const },
    { id: 6, title: 'Camera 006', detections: 0, isActive: true, color: 'yellow' as const },
  ];

  const handleViewStream = (streamId: number) => {
    console.log('View stream:', streamId);
    // Add navigation logic here
  };

  return (
    <div className="page-content">
      <PageHeader
        title="UniView - Surveillance Dashboard"
        subtitle="Real-time monitoring and tracking system"
      />

      <Grid>
        {mockStreams.map((stream) => (
          <StreamCard
            key={stream.id}
            id={stream.id}
            title={stream.title}
            detections={stream.detections}
            isActive={stream.isActive}
            color={stream.color}
            onViewStream={() => handleViewStream(stream.id)}
          />
        ))}
      </Grid>

      <InfoBox>
        <p>
          💡 <strong>Demo Mode:</strong> זהו דאשבורד דמו למעקב אחר אובייקטים.
          כל קופסה מייצגת מצלמה עם מספר האובייקטים שזוהו (TARGETS).
          כאשר תחבר את ה-API האמיתי, הנתונים יתעדכנו בזמן אמת עם מספר האנשים/אובייקטים שמזוהים בכל סטרים.
        </p>
      </InfoBox>
    </div>
  );
}

export default Home;

