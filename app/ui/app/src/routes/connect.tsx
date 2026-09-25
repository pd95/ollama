import { AppSidebar } from "@/components/AppSidebar";
import { ConnectAppsScreen } from "@/components/Onboarding";
import { SidebarLayout } from "@/components/layout/layout";
import { createFileRoute } from "@tanstack/react-router";
import { useEffect } from "react";
import { useSettings } from "@/hooks/useSettings";

export const Route = createFileRoute("/connect")({
  component: ConnectRoute,
});

function ConnectRoute() {
  const { settingsData, setSettings } = useSettings();

  useEffect(() => {
    if (!settingsData || settingsData.LastHomeView === "apps") return;
    setSettings({ LastHomeView: "apps" }).catch(() => {
      // Navigation must remain usable when best-effort persistence fails.
    });
  }, [settingsData, setSettings]);

  return (
    <SidebarLayout title="Apps" sidebar={<AppSidebar current="apps" />}>
      <ConnectAppsScreen />
    </SidebarLayout>
  );
}
