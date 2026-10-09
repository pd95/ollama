import { Children, type ReactElement, type ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { AppNavigation } from "./AppSidebar";
import { Route as IndexRoute } from "@/routes/index";

function startupRedirect(
  onboardingVersion: number,
  lastHomeView = "chat",
): Promise<unknown> {
  const beforeLoad = IndexRoute.options.beforeLoad;
  if (!beforeLoad) throw new Error("root route has no beforeLoad handler");

  return Promise.resolve(
    beforeLoad({
      context: {
        queryClient: {
          fetchQuery: vi.fn().mockResolvedValue({
            settings: {
              OnboardingVersion: onboardingVersion,
              LastHomeView: lastHomeView,
            },
          }),
        },
      },
    } as never),
  );
}

describe("desktop startup routing", () => {
  it("keeps onboarding for a fresh installation", async () => {
    await expect(startupRedirect(0)).rejects.toMatchObject({
      isRedirect: true,
      to: "/onboarding",
    });
  });

  it("opens a masked new chat for a returning installation", async () => {
    await expect(startupRedirect(1)).rejects.toMatchObject({
      isRedirect: true,
      to: "/c/$chatId",
      params: { chatId: "new" },
      mask: { to: "/" },
    });
  });

  it("restores Apps for a returning installation", async () => {
    await expect(startupRedirect(1, "apps")).rejects.toMatchObject({
      isRedirect: true,
      to: "/connect",
    });
  });

  it("keeps explicit Apps navigation on /connect", () => {
    const navigation = AppNavigation({ current: "chat" });
    const links = Children.toArray(navigation.props.children) as ReactElement<{
      to: string;
      children: ReactNode;
    }>[];
    const appLabel = Children.toArray(links[0].props.children)[1] as ReactElement<{
      children: ReactNode;
    }>;

    expect(links[0].props.to).toBe("/connect");
    expect(appLabel.props.children).toBe("Apps");
  });
});
